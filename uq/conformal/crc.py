import numpy as np
import torch

def get_lhat_crc(scores, alpha, B=1):
    n = len(scores)

    with torch.no_grad():

        # Set target alpha and exploratory lambdas - reduce search for computational efficiency around target 1-alpha.
        low_lim = np.quantile(scores, max(0,alpha-0.02))
        up_lim = np.quantile(scores, alpha+0.02)
        if up_lim-low_lim != 0:
            lambdas = np.arange(low_lim, up_lim, step=(up_lim-low_lim)/10)
        else:
            lambdas = [up_lim]
        
        # Compute loss for each potential lambda.
        rhat = []
        for ilambda in lambdas: 
            with torch.no_grad():
                rhat.append((1-(torch.tensor(scores) >= ilambda).to(torch.int16)).to(torch.float16).mean().item())
                torch.cuda.empty_cache()

        # Search threshold.
        lhat_idx = int(np.argmax(((n/(n+1)) * np.array(rhat) + B/(n+1)) >= (alpha)) - 1)
        return lambdas[lhat_idx]