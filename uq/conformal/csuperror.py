import numpy as np
import torch

def get_lhat_superror(pred_error, ref_error, alpha_psnr, precision=1e-5, reduce_search=True):
    with torch.no_grad():
        # Compute all posible thresholds.
        all_thresholds = np.concatenate([np.ravel(pred_error_i) for pred_error_i in pred_error])
        min_threshold = np.min(all_thresholds)
        max_threshold = np.max(all_thresholds)

        # Init risk control
        lower = min_threshold
        upper = max_threshold
        risk_lower = compute_conformal_risk(lower, pred_error, ref_error)
        risk_upper = compute_conformal_risk(upper, pred_error, ref_error)
        assert risk_lower <= risk_upper

        if psnr_sample(risk_upper) >= alpha_psnr:
            return upper
        else:
            while (upper - lower) > precision:
                middle = (lower + upper) / 2
                assert lower <= middle <= upper
                risk_middle = compute_conformal_risk(middle, pred_error, ref_error)
                if psnr_sample(risk_middle).item() >= alpha_psnr:
                    threshold = middle
                    lower = middle
                else:
                    upper = middle
            return threshold

def compute_conformal_risk(threshold, pred_error, ref_error):
    n = len(pred_error)
    risk_sum = 0.0
    for i in range(len(pred_error)):
        this_risk = 0.0
        risk_passed_threshold = ref_error[i][pred_error[i] <= threshold]
        if len(risk_passed_threshold) > 0:
            this_risk = max(this_risk, torch.max(risk_passed_threshold))
        risk_sum = risk_sum + this_risk
    return risk_sum / (n + 1) + 1 / (n + 1)

def psnr_sample(mse, max_psnr=48):
    return min(max_psnr, (10. * torch.log10(1. / (mse + 1e-8))))