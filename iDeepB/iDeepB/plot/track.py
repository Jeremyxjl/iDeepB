
import matplotlib.pyplot as plt

def plot_single_track(prediction, color='orange', alpha=0.3, figsizeSet =(6, 2) ):
    fig, ax = plt.subplots(figsize = figsizeSet)

    ax.fill_between(range(1, len(prediction) + 1), prediction, color = color, alpha= alpha)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('Position')
    ax.set_ylabel('Prediction')

    # picture_name = f'{task}_{row.chrom_TP}_{int(row.start_TP)}_{int(row.stop_TP)}'
    # plt.savefig(f'{args.output}_101bp_trackPlot/{picture_name}.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()