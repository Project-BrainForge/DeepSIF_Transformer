import numpy as np
from skimage.filters import threshold_otsu


def ispadding(x):
    # identify the padding in array
    return np.abs(x - 15213) < 1e-6


def get_otsu_regions(out, labels, args_params=None):
    """ Identify DeepSIF source region using otsu_threshold, run on CPU
    :param out: np.array; the output of DeepSIF, batch_size * num_time * num_region
    :param labels: np.array; group truth source region; batch_size * num_source * max_size; starts from 0
    :param args_params: optional parameters, could be
                        dis_matrix: np.array; distance between regions; num_region (994) * num_region
    :return return_eval: could be
                         all_regions: DeepSIF predicted regions; (batch_size, )
                         all_out:     DeepSIF predicted source activity;  (batch_size, )
    """
    # when there is no spike, the location error is nan

    batch_size = labels.shape[0]
    return_eval = dict()

    return_eval['all_regions'] = np.empty((batch_size,), dtype=object)
    return_eval['all_out'] = np.empty((batch_size,), dtype=object)

    for i in range(batch_size):
        thre_source = np.abs(out[i])
        thre_source = (thre_source - np.min(thre_source)) / np.max(thre_source)
        thresh = threshold_otsu(thre_source, nbins=100)
        select_pixel = out[i] > thresh
        otsu_region = np.where(np.sum(select_pixel, axis=0) > 7)[0]
        return_eval['all_regions'][i] = otsu_region
        return_eval['all_out'][i] = out[i, :, otsu_region]

    # Calculate the eval metrics in Python overall condition for all sources
    if args_params is not None:
        return_eval['precision'] = np.zeros(batch_size)
        return_eval['recall'] = np.zeros(batch_size)
        return_eval['le'] = np.zeros(batch_size)
        for i in range(batch_size):
            lb = labels[i][np.logical_not(ispadding(labels[i]))]
            recon = return_eval['all_regions'][i]
            overlap_region = len(np.intersect1d(lb, recon))
            # number of region based precision and recall
            return_eval['precision'][i] = overlap_region/len(recon) if len(recon) > 0 else 0
            return_eval['recall'][i] = overlap_region / len(lb) if len(lb) > 0 else 0
            if 'dis_matrix' in args_params and len(recon) > 0 and len(lb) > 0:
                le_each_region = np.min(args_params['dis_matrix'][recon][:, lb], axis=1)
                return_eval['le'][i] = np.mean(le_each_region)
            else:
                return_eval['le'][i] = np.nan

    return return_eval


def add_white_noise(sig, snr, args_params=None):
    """
    :param sig: np.array; num_electrode * num_time
    :param snr: int; signal to noise level in dB
    :param args_params: optional parameters, could be
                        ratio: np.array; ratio between white Gaussian noise and pre-set realistic noise
                        rndata: np.array; realistic noise data; num_sample * num_electrode * num_time
                        rnpower: np.array; pre-calculated power for rndata; num_sample * num_electrode

    :return: noise_sig: np.array; num_electrode * num_time
    """

    num_elec, num_time = sig.shape
    noise_sig = np.zeros((num_elec, num_time))
    sig_power = np.square(np.linalg.norm(sig, axis=1))/num_time
    if args_params is None:
        # Only add Gaussian noise
        for i in range(num_elec):
            noise_power = 10 ** (-(snr / 10)) * sig_power[i] / 2
            noise_std = np.sqrt(noise_power)
            noise_sig[i, :] = sig[i, :] + np.random.normal(0, noise_std, (num_time,))
    else:
        # Add realistic and Gaussian noise
        rnpower = args_params['rnpower']/num_time
        rndata = args_params['rndata']
        select_id = np.random.randint(0, rndata.shape[0])
        for i in range(num_elec):
            noise_power = 10 ** (-(snr / 10)) * sig_power[i]
            rpower = args_params['ratio']*noise_power                                 # realistic noise power
            noise_std = np.sqrt(noise_power - rpower)
            noise_sig[i, :] = sig[i, :] + np.random.normal(0, noise_std, (num_time,)) + np.sqrt(rpower/rnpower[select_id][i])*rndata[select_id][:, i]
    return noise_sig


def fwdJ_to_cortexJ(recon, rm):
    """
    :param recon: np.array; DeepSIF output, (num_time, num_region)
    :param rm: np.array; region mapping for each index, (num_vertices, )
    :return: J: np.array; DeepSIF output for each vertices, (num_time, num_vertices)
    """
    num_time, num_region = recon.shape
    num_vertices = rm.shape[0]
    J = np.zeros((num_time, num_vertices))
    for k in range(num_time):
        for i in range(num_region):
            J[k, rm==i] = recon[k, i]
    return J


def compute_correlation(pred, target):
    """
    Compute temporal correlation between predicted and target signals
    
    :param pred: np.array; predicted source signals (num_time, num_source)
    :param target: np.array; target source signals (num_time, num_source)
    :return: correlation: np.array; correlation for each source
    """
    correlations = []
    for i in range(pred.shape[1]):
        if np.std(pred[:, i]) > 1e-8 and np.std(target[:, i]) > 1e-8:
            corr = np.corrcoef(pred[:, i], target[:, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        else:
            correlations.append(0)
    return np.array(correlations)


def compute_localization_error(pred_regions, true_regions, distance_matrix=None):
    """
    Compute localization error between predicted and true source regions
    
    :param pred_regions: list; predicted source regions
    :param true_regions: list; true source regions  
    :param distance_matrix: np.array; distance matrix between regions (optional)
    :return: error: float; localization error
    """
    if len(pred_regions) == 0 or len(true_regions) == 0:
        return np.inf
        
    if distance_matrix is not None:
        # Use distance matrix to compute minimum distance
        distances = []
        for pred in pred_regions:
            min_dist = np.min([distance_matrix[pred, true] for true in true_regions])
            distances.append(min_dist)
        return np.mean(distances)
    else:
        # Use simple overlap-based metric
        overlap = len(set(pred_regions).intersection(set(true_regions)))
        return 1.0 - (overlap / len(set(pred_regions).union(set(true_regions))))


def normalize_data(data, method='minmax'):
    """
    Normalize data using different methods
    
    :param data: np.array; input data
    :param method: str; normalization method ('minmax', 'zscore', 'robust')
    :return: normalized_data: np.array
    """
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min > 1e-8:
            return (data - data_min) / (data_max - data_min)
        else:
            return data
    elif method == 'zscore':
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std > 1e-8:
            return (data - data_mean) / data_std
        else:
            return data - data_mean
    elif method == 'robust':
        data_median = np.median(data)
        data_mad = np.median(np.abs(data - data_median))
        if data_mad > 1e-8:
            return (data - data_median) / data_mad
        else:
            return data - data_median
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_synthetic_forward_matrix(num_sensors=75, num_sources=994, seed=42):
    """
    Create a synthetic forward matrix for testing purposes
    
    :param num_sensors: int; number of EEG sensors
    :param num_sources: int; number of source locations
    :param seed: int; random seed for reproducibility
    :return: forward_matrix: np.array; (num_sensors, num_sources)
    """
    np.random.seed(seed)
    
    # Create a realistic-looking forward matrix
    # Sources closer to sensors have higher values
    fwd = np.random.randn(num_sensors, num_sources) * 0.1
    
    # Add some structure to make it more realistic
    for i in range(num_sensors):
        # Each sensor is most sensitive to nearby sources
        center = int((i / num_sensors) * num_sources)
        width = num_sources // 10
        start = max(0, center - width)
        end = min(num_sources, center + width)
        fwd[i, start:end] *= 3
        
    return fwd
