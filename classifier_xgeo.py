
import xarray as xr
import xgeo # Needs to be imported to use geo extension

import geopandas as gpd
import gdal

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import logging
logger = logging.getLogger(__name__)

# A list of "random" colors
COLORS = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]


def dataframe_to_features(dataset_geo):
    features = dataset_geo.reset_index()
    
    # Return sampled pixel and its label for traning
    if 'class_id' in features.columns:
        X = features.image
        y = features.class_id
        
        return X, y
    
    # Return only the pixel value of each band for predection
    else:
        X = features.image
        return X


    
def sample_vectors_to_raster(vector_file_path, dataset_geo):
    """
    Rasterize, in a dataset, the vector file in the given path.
    The data of each file will be assigned the same pixel value. This value is defined by the order
    of the file in file_paths, starting with 1: so the points/poligons/etc in the same file will be
    marked as 1, those in the second file will be 2, and so on.
    :param vector_file_path: Path to a directory with shapefiles
    :param dataset_geo: Number of rows of the result
    :param cols: Number of columns of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    
    # Read the shape file from the given path
    shp = gpd.read_file(vector_file_path)
    
    # This line to cast the string in the shape file to int so xgeo would not give an error 
    shp["class_id"] = shp["class_id"].astype(int)
    
    # Sample the dataset with the given vector file
    sampled_labeled_raster = dataset_geo.geo.sample(vector_file=shp, value_name='class_id')
    
    # This to unstack the dataframe to be suitable for features
    df_labeled_raster = sampled_labeled_raster.unstack()
    
    """
    ## The pervious line proivde the following outputs which is super handy
    df_sampled
                                             image                        
    band                                         1     2     3     4     5
    class_id x             y            time                              
    1.0      597038.398110 9.547916e+06 0     56.0  23.0  17.0  64.0  61.0
                           9.547945e+06 0     59.0  23.0  19.0  75.0  54.0
                                            ...   ...   ...   ...   ...
    3.0      654437.221435 9.536488e+06 0     54.0  17.0  10.0   3.0   1.0
                       9.536516e+06 0     54.0  17.0  12.0   3.0   1.0
                       9.536545e+06 0     51.0  17.0  11.0   5.0   1.0
                                            ...   ...   ...   ...   ...                       
    """

    return df_labeled_raster

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
        raster_dataset = xr.open_rasterio(raster_data_path)
        raster_dataset = raster_dataset.to_dataset(name='image')
        geo_transform = raster_dataset.geo.transform
        proj = raster_dataset.geo.projection
        n_bands = len(raster_dataset.band)
        dim = ['x', 'y']
        rows, cols = [raster_dataset.sizes[xy] for xy in dim]
        
        df_raster = raster_dataset.to_dataframe().unstack("band")
        
    except RuntimeError as e:
        report_and_exit(str(e))

    logger.debug("Process the training data")\

    # create a dataframe from each pixel and its label
    df_labeled_pixels = sample_vectors_to_raster(train_data_path, raster_dataset)
    X_train, y_train = dataframe_to_features(df_labeled_pixels)
    n_class = y_train.unique()

    # Create dataframe each pixel of the image 
    X_predict = dataframe_to_features(df_raster)
    
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
    classifier.fit(X_train, y_train)

    logger.debug("Classifing...")
    result = classifier.predict(X_predict)

    # Reshape the result: split the labeled pixels into rows to create an image
    # need to figure out what is wrong with writegeotiff
    classification = result.reshape((rows, cols))
    write_geotiff(output_fname, n_class, classification, geo_transform, proj)
    logger.info("Classification created: %s", output_fname)

    #
    # Validate the results
    #
    if validation_data_path:
        logger.debug("Process the verification (testing) data")
        try:
            df_validation_pixels = sample_vectors_to_raster(validation_data_path, raster_dataset)
            X_test, y_test = dataframe_to_features(df_validation_pixels)

        except OSError.FileNotFoundError as e:
            report_and_exit(str(e))

        y_predicted = classifier.predict(X_test)

        logger.info("Confussion matrix:\n%s", str(
            metrics.confusion_matrix(y_test, y_predicted)))
        target_names = ['Class %s' % s for s in n_class]
        logger.info("Classification report:\n%s",
                    metrics.classification_report(y_test, y_predicted,
                                                  target_names=target_names))
        logger.info("Classification accuracy: %f",
                    metrics.accuracy_score(y_test, y_predicted))
        
        return y_test, y_predicted, classification