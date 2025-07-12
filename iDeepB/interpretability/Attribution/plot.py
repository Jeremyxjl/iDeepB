import logomaker
import pandas as pd
import numpy as np

# plot attribution by sequence
def plot_sequence_attribution(attribution_matrix, bases = ['A', 'C', 'G', 'T'], width=20, height=4):
    attribution_df = pd.DataFrame(attribution_matrix, columns=bases)

    # create Logo object
    attribution_logo = logomaker.Logo(attribution_df,
                            shade_below=.5,
                            fade_below=.5,
                            font_name='Arial Rounded MT Bold')

    # style using Logo methods
    attribution_logo.style_spines(visible=False)
    attribution_logo.style_spines(spines=['left', 'bottom'], visible=True)
    #attribution_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    attribution_logo.ax.set_ylabel("IG Attribution", labelpad=-1)
    #attribution_logo.ax.xaxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False)

    # adjust figure width and heigh
    attribution_logo.fig.set_figheight(height)
    attribution_logo.fig.set_figwidth(width)
    

def make_attribution_figure(contribution_score, ax):
    df = pd.DataFrame(contribution_score, columns=['A', 'C', 'G', 'U'])
    logo = logomaker.Logo(df, shade_below=0, fade_below=0, font_name='Arial Rounded MT Bold', ax=ax)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_xticks(np.arange(0, df.shape[0], 100))
    ax.set_xlim(0, df.shape[0])

    ax.set_yticks([])
    ax.set_ylim(np.min(contribution_score)-0.25, 1)

    ax.relim()
    ax.autoscale_view()

def make_attribution_logomaker(contribution_score, ax):
    df = pd.DataFrame(contribution_score, columns=['A', 'C', 'G', 'U'])
    # logo = logomaker.Logo(df, shade_below=0, fade_below=0, font_name='Arial Rounded MT Bold', ax=ax)
    logo = logomaker.Logo(df, shade_below=0, fade_below=0, ax=ax)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks([])
    ax.set_xticks(np.arange(0, df.shape[0], 100))
    
    # ax.set_xlim(0, df.shape[0])
    '''
    if np.max(contribution_score)>1:
        ax.set_ylim(np.min(contribution_score)-0.1, np.max(contribution_score))
    else:
        ax.set_ylim(np.min(contribution_score)-0.1, 1)
    '''
    # ax.set_ylim(-1, 1)
    # ax.set_yticks([0, 1])

    ax.relim()
    ax.autoscale_view()
'''
fig, ax = plt.subplots(figsize=(10, 1))
make_attribution_figure(contribution_score, ax)
plt.show()
'''