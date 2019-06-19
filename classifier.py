def add_one(x):
    x = x + 1
    return x


## function to write a geotiff for the classifed image output

def write_geotiff(fname, classes, data, geo_transform, projection, data_type=gdal.GDT_Byte):
    """
    Create a GeoTIFF file with the given data.
    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    # print(len(np.unique(data, return_inverse=True)))
    band.WriteArray(data)

    ct = gdal.ColorTable()
    for pixel_value in range(len(classes)+1):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Supervised classification.',
        'TIFFTAG_MAXSAMPLEVALUE': str(len(classes)),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    dataset = None  # Close the file
    return


def classic_classifier(method, raster_data_path, train_data_path, validation_data_path, output_fname):
    try:
        raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
    except RuntimeError as e:
        report_and_exit(str(e))

    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    for b in range(1, raster_dataset.RasterCount+1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())

    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape
    # A sample is a vector with all the bands data. Each pixel (independent of its position) is a
    # sample.
    n_samples = rows*cols

    logger.debug("Process the training data")
    try:
        files = [f for f in os.listdir(train_data_path) if f.endswith('.shp')]
        # print(files)
        classes = [f.split('.')[0] for f in files]
        shapefiles = [os.path.join(train_data_path, f) for f in files if f.endswith('.shp')]
    except OSError.FileNotFoundError as e:
        report_and_exit(str(e))

    labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
    is_train = np.nonzero(labeled_pixels)
    training_labels = labeled_pixels[is_train]
    training_samples = bands_data[is_train]

    flat_pixels = bands_data.reshape((n_samples, n_bands))

    #
    # Perform classification
    #
    CLASSIFIERS = {
        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'random-forest': RandomForestClassifier(n_jobs=4, n_estimators=10, class_weight='balanced'),
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        'svm': SVC(class_weight='balanced'),
        'linear-svm': SVC(kernel="linear", C=0.025),
        'svm-gamma': SVC(gamma=2, C=1),
        'nearest-neighbors': KNeighborsClassifier(3),
        'Gaussian-Process': GaussianProcessClassifier(1.0 * RBF(1.0)),
        'Decision-Tree': DecisionTreeClassifier(max_depth=5),
        'random-forest-1feature': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        'Neural-Net': MLPClassifier(alpha=1, max_iter=1000),
        'AdaBoost': AdaBoostClassifier(),
        'Naive-Bayes': GaussianNB(),
        'Quad-Discr': QuadraticDiscriminantAnalysis()
    }

    classifier = CLASSIFIERS[method]
    logger.debug("Train the classifier: %s", str(classifier))
    classifier.fit(training_samples, training_labels)

    logger.debug("Classifing...")
    result = classifier.predict(flat_pixels)

    # Reshape the result: split the labeled pixels into rows to create an image
    classification = result.reshape((rows, cols))
    write_geotiff(output_fname, classes, classification, geo_transform, proj)
    logger.info("Classification created: %s", output_fname)

    #
    # Validate the results
    #
    if validation_data_path:
        logger.debug("Process the verification (testing) data")
        try:
            shapefiles = [os.path.join(validation_data_path, "%s.shp" % c) for c in classes]
        except OSError.FileNotFoundError as e:
            report_and_exit(str(e))

        verification_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
        for_verification = np.nonzero(verification_pixels)
        verification_labels = verification_pixels[for_verification]
        predicted_labels = classification[for_verification]

        logger.info("Confussion matrix:\n%s", str(
            metrics.confusion_matrix(verification_labels, predicted_labels)))
        target_names = ['Class %s' % s for s in classes]
        logger.info("Classification report:\n%s",
                    metrics.classification_report(verification_labels, predicted_labels,
                                                  target_names=target_names))
        logger.info("Classification accuracy: %f",
                    metrics.accuracy_score(verification_labels, predicted_labels))
        return verification_labels, predicted_labels