import numpy as np
import pandas as pd
from IPython import embed
import sys
import configparser
import subprocess
import re
import matplotlib.pyplot as plt
import json
import pickle as pkl


def main(cfg_path):

    cp = configparser.ConfigParser()
    # option needed for case sensitivity
    cp.optionxform = str
    cp.read(cfg_path)

    inp_config  = cp['InputData']
    
    wgms_to_rgi_path = inp_config.get('wgms_to_rgi_path',fallback=None)
    wgms_id = inp_config.getint('glacier_wgms_id',fallback=None)
    parameter_sample_file_base = inp_config.get('parameter_sample_file_base',fallback=None)

    costwgt_str = inp_config.get('cost_variance_expansion',fallback='[1,1,1]')
    cost_wgts = np.array(json.loads(costwgt_str))

    dfwr = pd.read_csv(wgms_to_rgi_path)
    rgi_id = dfwr[dfwr.WGMS_ID == wgms_id].RGI60_ID.tolist()[0]
    analysis_csv = parameter_sample_file_base + '_' + rgi_id + '.csv'
    
    dfsample = pd.read_csv(analysis_csv,index_col=0)
    arr = dfsample.to_numpy()
    sample_arr = arr[:,:-3]
    results_arr = arr[:,-3:]
    param_names = dfsample.keys()[:-3]
    n_params = len(param_names)
    cost = np.sqrt(np.nansum((1/cost_wgts[None,:]**2) * results_arr**2,1)[:,None])
    
    with open("params_dan.ini", "r") as f:
        lines = f.readlines()
    with open("params_dan_" + str(wgms_id) + ".ini", "w") as f:
        for line in lines:
            line = re.sub(r'^.*glacier_wgms_id.*$', 'glacier_wgms_id = ' + str(wgms_id) + '\n', line)
            f.write(line)
    
    with open("params_dan.ini", "r") as f:
        lines = f.readlines()        
    with open("params_dan_" + str(wgms_id) + ".ini", "w") as f:
        for line in lines:            
            line = re.sub(r'^.*one_off_sample.*$', 'one_off_sample = -1\n', line)
            f.write(line)
    
    with open("params_dan.ini", "r") as f:
        lines = f.readlines()
    with open("params_dan_" + str(wgms_id) + ".ini", "w") as f:          
        for line in lines:     
            line = re.sub(r'^.*overwrite_sample_file.*$', 'overwrite_sample_file = False\n', line)
            f.write(line)

    stdout_file = open('stdout_' + str(wgms_id), 'w')
    stderr_file = open('stderr_' + str(wgms_id), 'w')
    proc = subprocess.Popen(["python", "fsm_sample_params.py", cfg_path], \
                stdout=stdout_file, stderr=stderr_file)
    
    prob = np.exp(-.5*cost**2)/np.sum(np.exp(-.5*cost**2));  # use a gaussian probability -- not sure what else to do

    probTensor = prob[:,:,None]
    
    cdf = np.cumsum(np.sort(prob)[::-1]);
    
    mean_param = np.sum(prob * sample_arr,0)
    cov_param = np.sum((sample_arr-mean_param)[:,:,None]*(sample_arr-mean_param)[:,None,:]*probTensor,0)
    
    
    sds = np.sqrt(cov_param.diagonal())
    
    sds_outer = sds[:,None]*sds[:,None].T
    corr_param = cov_param / sds_outer
    
    eival, ev = np.linalg.eig(cov_param)
    sort_ind = np.argsort(eival)[::-1]
    eival = eival[sort_ind]

    statsdict = {
	"mean": mean_param,
        "corr": corr_param,
        "cov": cov_param,
        "eigenvalues": eival,
        "eigenvectors": ev
    }
    fdump = open('stats_analysis_' + str(wgms_id) + '.pkl','wb')
    pkl.dump(statsdict,fdump)
    fdump.close()
    
    fig, axes = plt.subplots(n_params, n_params, figsize=(14, 14))
    
    bins=30
    cmap = 'viridis'
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:
                # Diagonal: weighted histogram
                ax.hist(
                    sample_arr[:, j],
                    bins=bins,
                    weights=prob.flatten(),
                    density=True,
                    color="steelblue",
                    alpha=0.8
                )
                
                text_str = f"{mean_param[j]:.3f} ± {sds[j]:.3f}"
                ax.text(
                    0.05, 0.9,
                    text_str,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                )
                ax.tick_params(axis='x', labelbottom=True)
                
            elif i<j:
                # Off-diagonal: weighted 2D histogram
                h = ax.hist2d(
                    sample_arr[:, j],
                    sample_arr[:, i],
                    bins=bins,
                    weights=prob.flatten(),
                    cmap=cmap
                )
                
            else:
                # Lower triangle: covariance text
                corr_ij = corr_param[i, j]
                ax.text(
                    0.5, 0.5,
                    f"{corr_ij:.3f}",
                    ha="center",
                    va="center",
                    fontsize=11
                )
                ax.set_xticks([])
                ax.set_yticks([])
                
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            

            if j == 0:
                ax.set_ylabel(param_names[i])
            else:
                ax.set_yticks([])
    plt.savefig('pair_plot_' + str(wgms_id) + '.png')
    proc.wait()
    stdout_file.close()
    stderr_file.close()


    original_file = cfg_path
    strs = cfg_path.split('.')
    new_file = strs[0] + '_' + str(wgms_id) + 'opt.' + strs[1]

    with open(original_file, 'r') as file:
        lines = file.readlines()

    # Modify lines with square brackets
    modified_lines = []
    for line in lines:
        # Check if the line contains square brackets
        if '[' in line and ']' in line:
            # Use regular expression to find the first number in brackets
            match = re.search(r'\[(\d+(?:\.\d+)?)', line)
            strs = line.split('=')
            parmname = strs[0].strip()
            if (parmname[:10]=='FSM_param_') & (match is not None):
                # Extract the string before the equal sign, the string before brackets, and the original line
                new_value = mean_param[np.where(param_names==parmname)[0][0]]

                original_line = line
                new_line = re.sub(r'\[(\d+\.\d+|\d+),', f'[{new_value},', line, count=1)
                modified_lines.append(new_line)
            else:
                modified_lines.append(line)  # If no match, append the original line
        else:
            modified_lines.append(line) 

    with open(new_file, 'w') as file:
        file.writelines(modified_lines)

    print(f"Mean values saved to {new_file}") 


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analyse_wgms_params.py <paramfile>")
        sys.exit(1)
    main(sys.argv[1])

