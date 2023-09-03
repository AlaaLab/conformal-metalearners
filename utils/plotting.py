from seaborn import ecdfplot
import matplotlib.pyplot as plt
import numpy as np


def plot_results(baseline_names, 
                 results_data,
                 oracle_width,
                 alpha=0.1,
                 save=False,
                 path=None, 
                 filename="test"):

    fig, axes   = plt.subplots(1, 3, figsize=(6.25, 3))

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.05,
                        hspace=0.4)

    # Box plot configurations
    boxprops    = dict(linestyle='-', linewidth=1, color="black", facecolor='lightyellow')
    medianprops = dict(linestyle='-', linewidth=.9, color='red')
    meanprops   = dict(marker='o', markersize=4.5, markeredgecolor='black', markerfacecolor='white')


    subplot_titles = ["Coverage", "Average Length", "RMSE"]

    for ax, data, title in zip(axes.flatten(), results_data, subplot_titles):
        ax.boxplot(data, 
                   boxprops=boxprops,
                   medianprops=medianprops,
                   meanprops=meanprops,
                   showmeans=True,
                   meanline=False,
                   showcaps=True,
                   showbox=True,
                   showfliers=True,
                   notch=False,
                   patch_artist=True,
                   widths=0.5,
                   vert=False,
                   flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': 'black', 'color': 'lightgray'})

        ax.tick_params(axis='both', which='major', labelsize=7.5)

        if title=="Coverage":
        
            ax.set_xlim(0, 1.1)
            ax.axvline(x=1-alpha, color='red', linewidth=3, alpha=0.25) 
            ax.set_yticklabels(baseline_names, rotation=0, ha='right')  # Set x-axis labels
            ax.set_xlabel("Coverage")
            ax.set_yticks(ticks=[1, 2, 3], labels=baseline_names)

        elif title=="Average Length":
            if oracle_width is not None:
                ax.axvline(x=oracle_width, color='blue', linewidth=3, alpha=0.25)
                ax.set_xlabel("Average Length")

        if title=="RMSE":
            ax.set_yticks([])
            ax.set_xlabel("RMSE")
        
    if save:
        plt.tight_layout()
        plt.savefig("figures/" + path + "/" + filename + ".png", dpi=3000, transparent=False)
    


def plot_stoch_dominance(cdf_x_, meta_scores, oracle_scores):

    plt.plot(cdf_x_, meta_scores[0], color="b")
    plt.fill_between(cdf_x_, meta_scores[1][:, 0], meta_scores[1][:, 1], alpha=0.1, color="b")

    plt.plot(cdf_x_, oracle_scores[0], color="r")
    plt.fill_between(cdf_x_, oracle_scores[1][:, 0], oracle_scores[1][:, 1], alpha=0.1, color="r")
    
    if np.max(np.abs(cdf_x_)) > 10 :
        
        plt.xlim(-10, 10)
    

def plot_stochastic_order(cdf_x, pseudo_scores, oracle_scores, 
                          save=False, path=None, filename="test"):
    
    fig, axes   = plt.subplots(1, 1, figsize=(1.5, 1))
    plot_stoch_dominance(cdf_x, pseudo_scores, oracle_scores)
    plt.yticks([0, 0.5, 1])
    axes.tick_params(axis='both', which='major', labelsize=7.5)
    
    if save:
        plt.tight_layout()
        plt.savefig("figures/" + path + "/" + filename + ".png", dpi=3000, transparent=False)
        

        
def plot_sweeps(alphas, results_dict, plot_params, path=None,  
                filename="test", save=True, calibration=True, 
                perf_metric="Coverage", alpha=.1):
    
    fig, axes   = plt.subplots(1, 1, figsize=(1.75, 1.5))
    metric_dict = dict({"Coverage": 0, 
                        "Average Length": 1, 
                        "RMSE": 2})

    for u in range(len(plot_params)):
        
        plt.plot(1-np.array(alphas), results_dict[:, metric_dict[perf_metric], u], **plot_params[u])

    if perf_metric=="Coverage":
        
        if calibration:
            
            plt.plot(1-np.array(alphas), 1-np.array(alphas), linestyle="--", linewidth=2, color="black")
    
        else:
            
            axes.axhline(y=1-alpha, linestyle="--", linewidth=2, color="black")

        plt.yticks([0, 0.5, 1])


    axes.tick_params(axis='both', which='major', labelsize=7.5)
    
    if save:
        plt.tight_layout()
        plt.savefig("figures/" + path + "/" + filename + ".png", dpi=3000, transparent=False)
