from pathlib import Path, PosixPath
import pandas as pd
import json
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Tuple, Dict
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

OPENML_DATASETS = ['6', '137', '151', '184', '1481', '41027', '43572',]
RENUVER_DATASETS = ['glass', 'bridges', 'restaurant', 'cars']

TEXTWIDTH_PT = 241.1474
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27
FIGURE_WIDTH = TEXTWIDTH_IN
FIGURE_HEIGHT = 16/9 * FIGURE_WIDTH

PATTERN_PALETTE = ['///', '...', '\\\\\\', '++', 'xx', 'OO']

ABLATION_BAR_CUSTOM_LABELS = {
    'all models': 'All Correctors',
    'llm_master': '(no) RD_ImpFM',
    'llm_correction': '(no) ET_CorrFM',
    'fd': '(no) SC_Phodi',
    'auto_instance': '(no) IM_DataWig',
}
ABLATION_SCATTER_CUSTOM_LABELS = {
    'all models': 'All Correctors',
    'auto_instance': 'IM_DataWig',
    'no auto_instance': 'no IM_DataWig',
    'llm_master': 'RD_ImpFM',
    'no llm_master': 'no RD_ImpFM',
    'fd': 'SC_Phodi',
    'no fd': 'no SC_Phodi',
    'llm_correction': 'ET_CorrFM',
    'no llm_correction': 'no ET_CorrFM',
} 
ABLATION_PATTERN_MAP = {
    'All Correctors': PATTERN_PALETTE[5],
    'IM_DataWig': PATTERN_PALETTE[1],
    'no IM_DataWig': PATTERN_PALETTE[1],
    'SC_Phodi': PATTERN_PALETTE[2],
    'no SC_Phodi': PATTERN_PALETTE[2], 
    'ET_CorrFM': PATTERN_PALETTE[3], 
    'no ET_CorrFM': PATTERN_PALETTE[3], 
    'RD_ImpFM': PATTERN_PALETTE[4],
    'no RD_ImpFM': PATTERN_PALETTE[4]
}


ACHSEN_FONTSIZE = 12
REST_FONTSIZE = ACHSEN_FONTSIZE - 3

ABLATION_COLUMN_ORDER_1 = list(reversed(['all models', 'no llm_master', 'no llm_correction',  'no fd', 'no auto_instance',]))
ABLATION_COLUMN_ORDER_2 = list(reversed(['all models', 'llm_master', 'llm_correction',  'fd', 'auto_instance',]))

ABLATION_COLOR_PALETTE = sns.color_palette("Set3", 6)

MIMIR_LABEL_COLORS = {
    'no auto_instance': ABLATION_COLOR_PALETTE[0],
    'no fd': ABLATION_COLOR_PALETTE[2],
    'no llm_correction': ABLATION_COLOR_PALETTE[3],
    'no llm_master': ABLATION_COLOR_PALETTE[4],
    'All Correctors': ABLATION_COLOR_PALETTE[5],
    'all models': ABLATION_COLOR_PALETTE[5],
    'auto_instance': ABLATION_COLOR_PALETTE[0],
    'fd': ABLATION_COLOR_PALETTE[2],
    'llm_correction': ABLATION_COLOR_PALETTE[3],
    'llm_master': ABLATION_COLOR_PALETTE[4],
    'IM_DataWig': ABLATION_COLOR_PALETTE[0],
    'no IM_DataWig': ABLATION_COLOR_PALETTE[0],
    'RD_ImpFM': ABLATION_COLOR_PALETTE[4],
    'no RD_ImpFM': ABLATION_COLOR_PALETTE[4],
    'SC_Phodi': ABLATION_COLOR_PALETTE[2],
    'no SC_Phodi': ABLATION_COLOR_PALETTE[2],
    'ET_CorrFM': ABLATION_COLOR_PALETTE[3],
    'no ET_CorrFM': ABLATION_COLOR_PALETTE[3],
    'f1_et_corr': ABLATION_COLOR_PALETTE[3],
    'value_model': '#999999', # baran value models
    'Ensembled Value Model': '#999999',
    'Ensembled SC_Phodi': ABLATION_COLOR_PALETTE[2],
    'Ensembled Vicinity Model': '#999999',
    'Ensembled llm_correction': ABLATION_COLOR_PALETTE[3],
    'Vicinity Model': '#999999',
}

def load_result(path_to_result: str):
    """
    Loads a ruska result and returns a tuple result_dict, config_dict. 
    """
    path = Path(path_to_result)
    config_flag = False
    result_flag = False

    result = ""
    result_config = ""

    with open(path, "rt") as f:
        for line in f:
            if line.strip() == "[BEGIN CONFIG]":
                config_flag = True
            elif line.strip() == "[END CONFIG]":
                config_flag = False
            elif line.strip() == "[BEGIN RESULTS]":
                result_flag = True
            elif line.strip() == "[END RESULTS]":
                result_flag = False
            else:
                if config_flag:
                    result_config = result_config + line
                elif result_flag:
                    result = result + line
    result_dict = eval(result)  # this is where PosixPath is used
    result_config_dict = eval(result_config)
    return result_dict, result_config_dict

def unify_measurements(results_list: List[Tuple[Path, dict]]) -> List[Dict]:
    res = []
    for path, additional_dimensions in results_list:
        r, _ = prepare_result_v1(load_result(path))
        r = [{**x, **additional_dimensions} for x in r]
        res.extend(r)
    return res

def prepare_result_v1(ruska_result: tuple):
    """
    Transform what is returned by Ruska.load_result() into a format that is
    more handy to work with. I use it for plotting, but it's more handy in other
    cases, too.
    First used in 2022W46.
    """
    result, ruska_config = ruska_result
    exp_config = ruska_config['config']
    full_result = [{**x['config'], **x['result']} for x in result]
    superfluous_config = [x for x in list(exp_config.keys()) if x not in ruska_config['ranges']]
    formatted_result = [{k: v for k, v in x.items() if k not in superfluous_config} for x in full_result]
    return formatted_result, ruska_config

def get_dataset_group(dataset: str):
    if dataset in ['beers', 'flights', 'rayyan', 'hospital']:
        return 'Baran'
    elif dataset in OPENML_DATASETS:
        return 'OpenML'
    elif dataset in RENUVER_DATASETS:
        return 'Renuver'


def format_feature_generators(feature_generators: list|str):
    feature_set = set(feature_generators)
    if feature_set == set(['fd', 'llm_correction', 'llm_master']):
        return 'no auto_instance'
    if feature_set == set(['auto_instance', 'fd', 'llm_correction', 'llm_master']):
        return 'all models'
    if feature_set == set(['auto_instance', 'llm_correction', 'llm_master']):
        return 'no fd'
    if feature_set == set(['auto_instance', 'fd', 'llm_master']):
        return 'no llm_correction'
    if feature_set == set(['auto_instance', 'fd', 'llm_correction']):
        return 'no llm_master'
    elif feature_set == set(['auto_instance']):
        return 'auto_instance'   
    elif feature_set == set(['vicinity']):
        return 'vicinity'    
    elif feature_set == set(['fd']):
        return 'fd'
    elif feature_set == set(['llm_correction']):
        return 'llm_correction'
    elif feature_set == set(['llm_master']):
        return 'llm_master'
    elif feature_set == set(['auto_instance', 'vicinity', 'llm_correction', 'llm_master']):
        return 'Ensembled Vicinity Model'
    elif feature_set == set(['value']):
        return 'value_model'
    elif feature_set == set(['auto_instance', 'fd', 'value', 'llm_master']):
        return 'Ensembled Value Model'
    else:
        raise ValueError()

def phodi_vs_vicinity_format_feature_generators(feature_generators):
    if feature_generators == 'all models':
        return 'Ensembled SC_Phodi'
    if feature_generators == 'fd':
        return 'SC_Phodi'
    if feature_generators == 'vicinity':
        return 'Vicinity Model'
    return feature_generators

def garf_experiment_metadata_from_filename(filename: str):
    """
    I decided for a stupid way of storing information about the GARF
    experiment in the filename. Here I pay for this.
    """
    splits = filename.split('_')
    name = splits[0]
    if len(splits) == 2:  # Baran
        error_class = None
        error_fraction = None
        timestamp = splits[1].split('.')[0]
    elif len(splits) == 4:  # Renuver
        error_class = None
        error_fraction = splits[1]
        timestamp = splits[3].split('.')[0]
    elif len(splits) == 5:
        error_class = f'{splits[1]}_{splits[2]}'
        error_fraction = splits[3]
        timestamp = splits[4].split('.')[0]
    elif len(splits) == 6:
        error_class = f'{splits[1]}_{splits[2]}_{splits[3]}'
        error_fraction = splits[4]
        timestamp = splits[5].split('.')[0]
    return {'dataset': name,
            'error_class': error_class,
            'error_fraction': error_fraction,
            'timestamp': timestamp}


def hc_experiment_metadata_from_filename(filename: str):
    """
    I decided for a stupid way of storing information about the HoloClean
    experiment in the filename. Here I pay for this.
    """
    splits = filename.split('_')
    if len(splits) == 4:  # Baran
        name = splits[0]
        name_dirty = splits[1] + '_' + splits[2]
        error_class = None
        error_fraction = None
        timestamp = splits[3].split('.')[0]
    elif len(splits) == 5:  # RENUVER
        name = splits[0]
        name_dirty = splits[1]
        error_fraction = splits[2]
        error_class = None
        timestamp = splits[4].split('.')[0]
    elif len(splits) == 6:  # OpenML simple_mcar
        name = splits[0]
        name_dirty = splits[1]
        error_class = splits[2] + '_' + splits[3]
        error_fraction = splits[4]
        timestamp = splits[5].split('.')[0]
    elif len(splits) == 7:  # OpenML imputer_simmple_mcar
        name = splits[0]
        name_dirty = splits[1]
        error_class = splits[2] + '_' + splits[3] + '_' + splits[4]
        error_fraction = splits[5]
        timestamp = splits[6].split('.')[0]
    else:
        raise ValueError('wtf')
    return {'dataset': name, 
            'dataset_dirty': name_dirty, 
            'error_fraction': error_fraction, 
            'error_class': error_class, 
            'timestamp': timestamp}

def normalize_dataset(row):
    if row['error_fraction'] == '':
        return row['dataset']
    elif row['dataset'] in ['beers', 'flights', 'hospital', 'rayyan']:
        return row['dataset']
    elif (row.get('error_class', '') == '') or (row.get('error_class', '') is None):
        return f"{row['dataset']} {int(row['error_fraction'])}%"
    elif row.get('error_class', '').startswith('imputer'):
        return f"{row['dataset']} \n cat {int(row['error_fraction'])}%"
    return f"{row['dataset']} {int(row['error_fraction'])}%"

def get_ablation_study(ablation_study_dir: str):
    measurements = []
    pathlist = Path(ablation_study_dir).glob('*.json')
    for path in pathlist:
        with open(path) as f:
            measurements.append(json.load(f))
    
    failed_measurements = [m for m in measurements if m['status'] == 0]
    print(f'Loaded Ablation Study. {len(failed_measurements)}/{len(measurements)} measurements failed.')
    results = [{**m['config'], **m['result']} for m in measurements if m['status'] == 1]  # some measurements crash
    
    # filter out openml-models with 1% errors, because they're always perfectly cleaned
    results = [r for r in results if not (r['dataset'] in OPENML_DATASETS and str(r['error_fraction']) == '1')]
    
    # make feature generators lisible
    results = [{**r, 'feature_generators': format_feature_generators(r.get('feature_generators'))} for r in results]
    
    return pd.DataFrame(results), failed_measurements

def plot_global_ablation_study(ablation_study_dir: str) -> Tuple[plt.figure, Tuple[plt.axes, plt.axes], List]:
    """
    Big fat ablation study on all datasets.
    """

    df, failed_measurements = get_ablation_study(ablation_study_dir)

    df = (df.loc[:, ['dataset', 'feature_generators', 'error_class', 'error_fraction', 'f1']]
        .groupby(['dataset', 'feature_generators', 'error_class', 'error_fraction'])
        .agg({'f1': 'mean', })
        .reset_index()
        )

    df['normal_dataset'] = df.apply(normalize_dataset, axis=1)
    pivot_df = df.pivot(index='normal_dataset', columns='feature_generators', values='f1')

    # Reorder the columns of the pivot_df dataframe for the first sub-plot
    sub_pivot_df1 = pivot_df[ABLATION_COLUMN_ORDER_1]

    # Create a figure and two sub-plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*2, FIGURE_HEIGHT), sharey=True)

    # Plot the first sub-plot
    sub_pivot_df1.plot(kind='barh', ax=ax1, color=[MIMIR_LABEL_COLORS[label] for label in sub_pivot_df1.columns], legend=False, width=0.65)
    ax1.set_ylabel('')
    ax1.set_xlabel('$F_1$ score')
    ax1.set_title('Drop One Corrector')

    # Reorder the columns of the pivot_df dataframe for the second sub-plot
    sub_pivot_df2 = pivot_df[ABLATION_COLUMN_ORDER_2]

    # Plot the second sub-plot
    sub_pivot_df2.plot(kind='barh', ax=ax2, color=[MIMIR_LABEL_COLORS[label] for label in sub_pivot_df2.columns], legend=False, width=0.65)
    ax2.set_ylabel('Dataset')
    ax2.set_xlabel('$F_1$ score')
    ax2.set_title('Select One Corrector')

    # Create custom handles for the legend
    custom_handles = [mpatches.Patch(color=MIMIR_LABEL_COLORS[key], label=ABLATION_BAR_CUSTOM_LABELS[key]) for key in ABLATION_BAR_CUSTOM_LABELS]

    fig.legend(custom_handles, ABLATION_BAR_CUSTOM_LABELS.values(), title='Ensembles', loc='upper right', fontsize=REST_FONTSIZE, bbox_to_anchor=(1, 1))

    # Adjust layout and save the figure
    plt.tight_layout(rect=(0, 0, 0.88, 1.06))
    return fig, (ax1, ax2), failed_measurements

def ablation_scatter(mimir_ablation_path: str) -> Tuple[plt.figure, plt.axes, List]:
    results, failed_measurements = get_mimir_result(mimir_ablation_path)
    results = [{**r, 'feature_generators': format_feature_generators(r['feature_generators'])} for r in results]
    df = pd.DataFrame(results)
    df['normal_dataset'] = df.apply(normalize_dataset, axis=1)

    df = (df.loc[:, ['normal_dataset', 'feature_generators', 'error_class', 'error_fraction', 'f1']]
        .groupby(['normal_dataset', 'feature_generators', 'error_class', 'error_fraction'])
        .agg({'f1': 'mean'})
        .reset_index()
    )

    pivot_df = df.pivot(index='normal_dataset', columns='feature_generators', values='f1')

    # Plotting with Seaborn
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Plotting other models
    for col in pivot_df.columns:
        if col != 'all models':
            sns.scatterplot(x=pivot_df[col], y=pivot_df['all models'], label=col, marker='.', ax=ax)

    # Plotting the diagonal line
    ax.plot([0, 1], [0, 1], '-k', alpha=.5)
    ax.set_aspect('equal', adjustable='box')

    # Adjust legend position
    handles, _ = ax.get_legend_handles_labels()
    labels = [ABLATION_SCATTER_CUSTOM_LABELS[c] for c in pivot_df.columns if c != 'all models']
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=True, title='Ablations')

    # Set axis labels
    plt.xlabel("$F_1$ score Ablations")
    plt.ylabel("$F_1$ score Mimir's Ensemble")

    return fig, ax, failed_measurements


def plot_joint_ablation(ablation_study_dir: str) -> Tuple[plt.figure, Tuple[plt.axes, plt.axes], List]:
    """
    Joint local ablation barchat and scatterplot.
    """
    df_global, failed_measurements = get_ablation_study(ablation_study_dir)
    df_global = df_global.groupby(['dataset', 'feature_generators', 'error_fraction', 'error_class']).agg({'f1': 'mean',
                                                                                        'precision': 'mean',
                                                                                        'recall': 'mean',
                                                                                        'run': list}).reset_index()
    df_global['normal_dataset'] = df_global.apply(normalize_dataset, axis=1)

    # get relevant datasets
    df = df_global[df_global['normal_dataset'].isin(['flights', 'beers', '151 \n cat 5%', '43572 \n cat 5%'])]

    pivot_df = df.pivot(index='normal_dataset', columns='feature_generators', values='f1')

    # Reorder the columns of the pivot_df dataframe for the first sub-plot
    sub_pivot_df1 = pivot_df[ABLATION_COLUMN_ORDER_1]

    fig = plt.figure(constrained_layout=True)
    gs0 = gridspec.GridSpec(2, 2, figure=fig)

    gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[2:])
    ax0 = fig.add_subplot(gs00[:, 0])
    ax1 = fig.add_subplot(gs00[:, 1])

    gs10 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0:2])
    ax2 = fig.add_subplot(gs10[:,1])

    # Plot the first sub-plot
    sub_pivot_df1.plot(kind='barh', ax=ax0, rot=0, color=[MIMIR_LABEL_COLORS[label] for label in sub_pivot_df1.columns], legend=False, width=0.75)
    ax0.set_ylabel('', labelpad=4, fontsize=ACHSEN_FONTSIZE)
    ax0.set_xlabel('$F_1$ score', labelpad=-12, fontsize=ACHSEN_FONTSIZE)
    ax0.set_title('')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    # Define headlines for barcharts
    h_headlines = [plt.plot([],marker=m, ls="", color='black')[0] for m in ['X', 'o']]
    l_headlines = ["Drop One Corrector", "Select One Corrector"]

    # improvise a headline with the legend - forgive me.
    leg = ax0.legend(
               [h_headlines[0]],
               [l_headlines[0]],
               loc='upper center',
               bbox_to_anchor=(.50, 1.15),
               borderaxespad=0.,
               ncol=1,
               frameon=False,
               fontsize=ACHSEN_FONTSIZE)

    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(8)

    ax0.set_xticks([0.0, 1.0])  # Set x-axis ticks

    # Remove ticks from the y-axis
    ax0.tick_params(axis='y', which='both', left=False, labelleft=True, pad=0, labelsize=ACHSEN_FONTSIZE)

    for i, b in enumerate(ax0.patches):
        b.set_hatch(PATTERN_PALETTE[i//4])
        b.set_edgecolor("black")
        
    # Reorder the columns of the pivot_df dataframe for the second sub-plot
    sub_pivot_df2 = pivot_df[ABLATION_COLUMN_ORDER_2]

    # Plot the second sub-plot
    sub_pivot_df2.plot(kind='barh', ax=ax1, rot=0, color=[MIMIR_LABEL_COLORS[label] for label in sub_pivot_df2.columns], legend=False, width=0.75)
    ax1.set_xlabel('$F_1$ score', labelpad=-12, fontsize=ACHSEN_FONTSIZE)
    ax1.set_ylabel('')
    ax1.set_title('')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_xticks([0.0, 1.0])

    # improvise a headline with the legend - forgive me.
    leg = ax1.legend(
               [h_headlines[1]],
               [l_headlines[1]],
               loc='upper center',
               bbox_to_anchor=(.50, 1.15),
               borderaxespad=0.,
               ncol=1,
               frameon=False,
               fontsize=ACHSEN_FONTSIZE)

    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(8)

    # Remove ticks from the y-axis
    ax1.tick_params(axis='y', which='both', left=False, labelleft=False)

    for i, b in enumerate(ax1.patches):
        b.set_hatch(PATTERN_PALETTE[i//4])
        b.set_edgecolor("black")

    # Plot scatters at the bottom
    results, failed_measurements = get_mimir_result(ablation_study_dir)
    results = [{**r, 'feature_generators': format_feature_generators(r['feature_generators'])} for r in results]
    df_scatter = pd.DataFrame(results)
    df_scatter['normal_dataset'] = df_scatter.apply(normalize_dataset, axis=1)

    df_scatter = (df_scatter.loc[:, ['normal_dataset', 'feature_generators', 'error_class', 'error_fraction', 'f1']]
        .groupby(['normal_dataset', 'feature_generators', 'error_class', 'error_fraction'])
        .agg({'f1': 'mean'})
        .reset_index()
    )

    pivot_df = df_scatter.pivot(index='normal_dataset', columns='feature_generators', values='f1')

    for col in pivot_df.columns:
        if col != 'all models':
            marker = 'X' if col.startswith('no') else 'o'
            sns.scatterplot(x=pivot_df[col], y=pivot_df['all models'], color=MIMIR_LABEL_COLORS[col], label=col, marker=marker, ax=ax2, legend=False)

    # Plotting the diagonal line
    ax2.plot([0, 1], [0, 1], '-k', alpha=.5)
    ax2.set_aspect('equal', adjustable='box')

    h_select, l_select = ax1.get_legend_handles_labels()
    l_select = [ABLATION_BAR_CUSTOM_LABELS[c] for c in l_select]

    handles = [*reversed(h_select)]
    labels = [*reversed(l_select)]

    leg = fig.legend(
               handles,
               labels,
               loc='center left',
               bbox_to_anchor=(0.17, .8),
               borderaxespad=0.,
               ncol=1,
               frameon=True,
               title='Ablations',
               title_fontsize=ACHSEN_FONTSIZE)

    # Set axis labels
    ax2.set_xlabel("$F_1$ score Ablations", fontsize=ACHSEN_FONTSIZE)
    ax2.set_ylabel("$F_1$ score Mimir", fontsize=ACHSEN_FONTSIZE)

    ax2.set_xticks([0.0, .2, .4, .6, .8, 1.0])
    ax2.set_yticks([0.0, .2, .4, .6, .8, 1.0])
    
    arrow = patches.ConnectionPatch(
        [0.95, 2.95],
        [1, 0.84],
        coordsA=ax1.transData,
        coordsB=ax2.transData,
        connectionstyle=patches.ConnectionStyle("angle3", ),#rad=-0.4),
        color="black",
        arrowstyle="<-",  # "normal" arrow
        mutation_scale=20,  # controls arrow head size
        linewidth=1.5,
    )
    fig.patches.append(arrow)

    return fig, (ax0, ax1, ax2), failed_measurements, pivot_df

def plot_local_ablation_study(ablation_study_dir: str) -> Tuple[plt.figure, Tuple[plt.axes, plt.axes], List]:
    """
    Local ablation study to showcase corrector's properties.
    """
    df_global, failed_measurements = get_ablation_study(ablation_study_dir)
    df_global = df_global.groupby(['dataset', 'feature_generators', 'error_fraction', 'error_class']).agg({'f1': 'mean',
                                                                                        'precision': 'mean',
                                                                                        'recall': 'mean',
                                                                                        'run': list}).reset_index()
    df_global['normal_dataset'] = df_global.apply(normalize_dataset, axis=1)

    # get relevant datasets
    df = df_global[df_global['normal_dataset'].isin(['flights', 'beers', '151 \n cat 5%', '43572 \n cat 5%'])]

    pivot_df = df.pivot(index='normal_dataset', columns='feature_generators', values='f1')

    # Reorder the columns of the pivot_df dataframe for the first sub-plot
    sub_pivot_df1 = pivot_df[ABLATION_COLUMN_ORDER_1]

    # Create a figure and two sub-plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*2, FIGURE_HEIGHT), sharey=True,)

    # Plot the first sub-plot
    sub_pivot_df1.plot(kind='barh', ax=ax1, rot=0, color=[MIMIR_LABEL_COLORS[label] for label in sub_pivot_df1.columns], legend=False, width=0.75)
    ax1.set_ylabel('Dataset', labelpad=4, fontsize=ACHSEN_FONTSIZE)
    ax1.set_xlabel('$F_1$ score', labelpad=-8, fontsize=ACHSEN_FONTSIZE)
    ax1.set_title('Drop One Corrector')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_xticks([0.0, 1.0])  # Set x-axis ticks

    # Remove ticks from the y-axis
    ax1.tick_params(axis='y', which='both', left=False, labelleft=True, pad=0, labelsize=ACHSEN_FONTSIZE)

    for i, b in enumerate(ax1.patches):
        b.set_hatch(PATTERN_PALETTE[i//4])
        #b.set_rasterized(True)
        b.set_edgecolor("black")
        
    # Reorder the columns of the pivot_df dataframe for the second sub-plot
    sub_pivot_df2 = pivot_df[ABLATION_COLUMN_ORDER_2]

    # Plot the second sub-plot
    sub_pivot_df2.plot(kind='barh', ax=ax2, rot=0, color=[MIMIR_LABEL_COLORS[label] for label in sub_pivot_df2.columns], legend=False, width=0.75)
    ax2.set_xlabel('$F_1$ score', labelpad=-8, fontsize=ACHSEN_FONTSIZE)
    ax2.set_title('Select One Corrector')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_xticks([0.0, 1.0])  # Set x-axis ticks

    # Remove ticks from the y-axis
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False)

    for i, b in enumerate(ax2.patches):
        b.set_hatch(PATTERN_PALETTE[i//4])
        #b.set_rasterized(True)
        b.set_edgecolor("black")

    # Create custom handles for the legend
    custom_handles = [mpatches.Patch(facecolor=MIMIR_LABEL_COLORS[key], 
                                    label=ABLATION_BAR_CUSTOM_LABELS[key], 
                                    hatch=list(reversed(PATTERN_PALETTE))[i], 
                                    #rasterized=True,
                                    edgecolor='black') for i, key in enumerate(ABLATION_BAR_CUSTOM_LABELS)]

    fig.legend(custom_handles, ABLATION_BAR_CUSTOM_LABELS.values(), title='Correctors', 
            title_fontsize=ACHSEN_FONTSIZE, loc='lower center', fontsize=REST_FONTSIZE,
            bbox_to_anchor=(0.5, -0.15, 0, 0), ncol=len(ABLATION_BAR_CUSTOM_LABELS)//2, frameon=False)

    return fig, (ax1, ax2), failed_measurements 

def get_baran_experiment(baran_results_path: str, baran_experiment_name: str) -> List[dict]:
    """
    Global performance of Baran on all datasets.
    """
    res_baran = unify_measurements([
    (str(Path(baran_results_path) / baran_experiment_name) + '-baran.txt', {'dataset_group': 'Baran', 'ensemble': 'Baran'}),
    (str(Path(baran_results_path) / baran_experiment_name) + '-openml.txt', {'dataset_group': 'OpenML', 'ensemble': 'Baran'}),
    (str(Path(baran_results_path) / baran_experiment_name) + '-renuver.txt', {'dataset_group': 'Renuver', 'ensemble': 'Baran'}),
    ])

    # set default values for error_fraction and error_class to enable grouping
    res_baran = [{'error_fraction': '', 'error_class': '', **r} 
                if r['dataset_group'] != 'OpenML'
                else {'error_fraction': 5, **r} for r in res_baran]

    # filter out 1% on OpenML because these are always perfectly cleaned
    res_baran = [r for r in res_baran if not (r['dataset'] in OPENML_DATASETS and str(r['error_fraction']) == '1')]

    # normalize dataset names
    res_baran = [{**r, 'normalized_dataset': normalize_dataset(r)} for r in res_baran]
    return res_baran

def get_mimir_result(mimir_results_path: str) -> Tuple[List[dict], List]:
    res_mimir = []
    pathlist = Path(mimir_results_path).glob('*.json')
    for path in pathlist:
        with open(path) as f:
            res_mimir.append(json.load(f))

    failed_measurements = [m for m in res_mimir if m['status'] == 0]
    print(f'Loaded Mimir Results. {len(failed_measurements)}/{len(res_mimir)} measurements failed.')

    res_mimir = [{**m['config'], **m['result'], 'ensemble': 'Mimir', 'dataset_group': get_dataset_group(m['config']['dataset'])} for m in res_mimir if m['status'] == 1]  # some measurements crash
    res_mimir = [m for m in res_mimir if not (m['dataset_group'] == 'OpenML' and m['error_fraction'] == 1)]
    res_mimir = [{**m, 'normalized_dataset': normalize_dataset(m)} for m in res_mimir]

    # filter out 1% on OpenML because these are always perfectly cleaned
    res_mimir = [r for r in res_mimir if not (r['dataset'] in OPENML_DATASETS and str(r['error_fraction']) == '1')]
    return res_mimir, failed_measurements

def get_holoclean_global_performance(holoclean_results_path: str):
    hc_results = []
    hc_results_dir = Path(holoclean_results_path)
    pathlist = Path(hc_results_dir).glob('*.txt')
    for path in pathlist:
        with open(path) as f:
            measurement = json.load(f)
            metadata = hc_experiment_metadata_from_filename(path.name)
            hc_results.append({**measurement, **metadata})

    hc_results = [{**m, 'ensemble': 'HoloClean'} for m in hc_results]
    hc_results = [{'normalized_dataset': normalize_dataset(m), **m} for m in hc_results]

    # filter out 1% on OpenML because these are always perfectly cleaned
    hc_results = [r for r in hc_results if not (r['dataset'] in OPENML_DATASETS and str(r['error_fraction']) == '1')]

    # Filter out 2%, 4%, 5% on RENUVER because others don't measure it.
    hc_results = [r for r in hc_results if not (r['dataset'] in RENUVER_DATASETS and str(r['error_fraction']) in ['2', '4', '5'])]
    return hc_results

def get_garf_global_performance(garf_results_path: str):
    results = []
    garf_results_dir = Path(garf_results_path)
    pathlist = Path(garf_results_dir).glob('*.txt')
    for path in pathlist:
        with open(path, 'rt') as f:
            measurement = json.load(f)
            metadata = garf_experiment_metadata_from_filename(path.name)
            results.append({**measurement, **metadata,})
    results = [{'normalized_dataset': normalize_dataset(r),
                'dataset': r['dataset'],
                'ensemble': 'Garf',
                'ed_p': r.get('ed_p'),
                'ed_r': r.get('ed_r'),
                'ed_f': r.get('ed_f'),
                'precision': r.get('et_p'),
                'recall': r.get('et_r'),
                'f1': r.get('et_f'),
                'error_class': r.get('error_class'),
                'error_fraction': r.get('error_fraction')} for r in results] 

    # filter out 1% on OpenML because these are always perfectly cleaned
    # and 10% on OpenML because others don't measure it.
    results = [r for r in results if not (r['dataset'] in OPENML_DATASETS and str(r['error_fraction']) in ['1', '10'])]
    
    # Filter out 2%, 4%, 5% on RENUVER because others don't measure it.
    results = [r for r in results if not (r['dataset'] in RENUVER_DATASETS and str(r['error_fraction']) in ['2', '4', '5'])]
    return results

def plot_mimir_vs_baran(mimir_results_path: str, baran_results_path: str, baran_experiment_name: str) -> Tuple[plt.figure, plt.axes, pd.DataFrame, List]:
    """
    Boxplot global performance Mimir vs Baran.
    """
    res_baran = get_baran_experiment(baran_results_path, baran_experiment_name)
    res_mimir, failed_measurements = get_mimir_result(mimir_results_path)

    df = pd.DataFrame([*res_mimir, *res_baran])

    df_melt = pd.melt(df, id_vars=['dataset_group', 'ensemble'], value_vars=['f1'], 
                  var_name='F1 Type', value_name='F1 Score')

    fig, ax = plt.subplots(figsize=(FIGURE_HEIGHT, FIGURE_WIDTH))

    # Create the boxplot
    sns.boxplot(data=df_melt, x='dataset_group', y='F1 Score', hue='ensemble', 
                    palette='Set3', fliersize=2, linewidth=0.8, showfliers=False, ax=ax)

    # Set axis labels and title
    ax.set_xlabel('')
    ax.set_ylabel('$F_1$ score', fontsize=ACHSEN_FONTSIZE)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='x', which='both', bottom=False, labelsize=ACHSEN_FONTSIZE)

    # Set y-axis ticks
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # Add legend
    ax.legend(loc='best',
            #bbox_to_anchor=(0.5, 1.1), 
            fontsize=REST_FONTSIZE, 
            ncol=len(df_melt['ensemble'].unique())
            )

    # Limit the y-axis from 0 to 1
    ax.set_ylim([0, 1.05])

    return fig, ax, df_melt, failed_measurements

def performance_table(mimir_results_path: str, baran_results_path: str, baran_experiment_name: str, garf_results_path: str, holoclean_results_path: str) -> Tuple[pd.DataFrame, List]:
    res_mimir, failed_measurements = get_mimir_result(mimir_results_path)
    res_mimir = [{'normalized_dataset': r['normalized_dataset'],
                   'ensemble': r['ensemble'],
                   'f1': r['f1'],
                   'precision': r['precision'],
                   'recall': r['recall']} for r in res_mimir]

    res_baran = get_baran_experiment(baran_results_path, baran_experiment_name)
    res_baran = [{'normalized_dataset': r['normalized_dataset'],
                   'ensemble': r['ensemble'],
                   'f1': r['f1'],
                   'precision': r['precision'],
                   'recall': r['recall']} for r in res_baran]

    res_hc = get_holoclean_global_performance(holoclean_results_path)
    res_hc = [{'normalized_dataset': r['normalized_dataset'],
                   'ensemble': r['ensemble'],
                   'f1': r['f1'],
                   'precision': r['precision'],
                   'recall': r['recall']} for r in res_hc]

    res_garf = get_garf_global_performance(garf_results_path)
    res_garf = [{'normalized_dataset': r['normalized_dataset'],
                   'ensemble': r['ensemble'],
                   'f1': r['f1'],
                   'precision': r['precision'],
                   'recall': r['recall']} for r in res_garf]
    df = pd.DataFrame([*res_mimir, *res_baran, *res_hc, *res_garf])

    return (pd.pivot((df.loc[:, ['normalized_dataset', 'ensemble', 'f1']]
            .groupby(['normalized_dataset', 'ensemble'])
            .agg({'f1': 'mean'})
            .reset_index()),
            columns='ensemble',
            index='normalized_dataset',
            values='f1')
            .round(2)
           ), failed_measurements

def et_corrfm_vs_value_model_v2(mimir_result_dir: str) -> Tuple[plt.figure, List[plt.axes], pd.DataFrame]:
    """
    Neue Version des Plots, in der das Value Model in Mimir implementiert ist.
    """
    res_mimir, failed_measurements = get_mimir_result(mimir_result_dir)
    res_mimir = [{'normalized_dataset': r['normalized_dataset'],
                   'feature_generators': format_feature_generators(r['feature_generators']),
                   'f1': r['f1'],
                   'precision': r['precision'],
                   'recall': r['recall']} for r in res_mimir]
    df = (pd.DataFrame(res_mimir)
                .groupby(['normalized_dataset', 'feature_generators'])
                .agg({'f1': 'mean',
                    'precision': 'mean',
                    'recall': 'mean',
                    })
                .reset_index())
    dataset_subset = ['beers', 'flights', 'hospital', 'rayyan', 'cars 1%', 'cars 3%', 'glass 1%', 'glass 3%', ]
    df_scatter = df.pivot(values=['f1', 'precision', 'recall'], index=['normalized_dataset'], columns=['feature_generators'])
    df_scatter_subset = df_scatter[df_scatter.index.isin(dataset_subset)]

    df_bars = df[df['normalized_dataset'].isin(dataset_subset)]
    df_bars = df_bars[df_bars['feature_generators'].isin(['llm_correction', 'value_model'])]

    fig = plt.figure(constrained_layout=True)
    axes = fig.subplot_mosaic([[0, 1],[2, 2]])
    palette = sns.color_palette('Set3')

    # Remove spines
    sns.despine()

    for i, measure, label in [(0, 'precision', 'Precision'), (1, 'recall', 'Recall')]:
        sns.scatterplot(x=df_scatter[measure]['value_model'], y=df_scatter[measure]['llm_correction'], ax=axes[i], color=palette[4])
        sns.lineplot(x=[0, 1], y=[0, 1], ax=axes[i], color='black', alpha=.5)
        
        axes[i].set_xlabel(f'{label} Value Model', fontsize=ACHSEN_FONTSIZE)
        axes[i].set_ylabel(f'{label} ET_CorrFM', fontsize=ACHSEN_FONTSIZE)
        axes[i].set_xticks([0, 0.25, 0.5, 0.75, 1.00])
        axes[i].set_yticks([0, 0.25, 0.5, 0.75, 1.00])
        axes[i].set_aspect('equal', adjustable='box')
        axes[i].set_xticklabels([0, '', '', '', 1.00], fontsize=REST_FONTSIZE)
        axes[i].set_yticklabels([0, '', '', '', 1.00], fontsize=REST_FONTSIZE)

        # Plot highlighted datasets
        axes[i].scatter(x=df_scatter_subset[measure]['value_model'], y=df_scatter_subset[measure]['llm_correction'], color='black')

        # Annotate the scatter plot for 'hospital' dataset
        for d in dataset_subset:
            x_value, y_value = df_scatter[df_scatter.index == d][measure]['value_model'], df_scatter[df_scatter.index == d][measure]['llm_correction']
            axes[i].annotate(d, (x_value, y_value), textcoords="offset points", xytext=(10, 10), ha="right", va="top", fontsize=8)

    df['feature_generators'] = df['feature_generators'].replace('all models', 'Ensembled llm_correction')
    sns.boxplot(y='feature_generators', x='f1', data=df, ax=axes[2], palette=MIMIR_LABEL_COLORS)
    custom_labels = ['Ensembled Value Model', 'Ensembled ET_CorrFM', 'ET_CorrFM', 'Value Model']  # Replace with your desired labels
    axes[2].set_yticklabels(custom_labels)
    # sns.swarmplot(data=df[df['normalized_dataset'].isin(dataset_subset)], x='f1', y='feature_generators', color='black', size=3.5, dodge=True, legend=False, ax=axes[2])

    axes[2].set_xlabel('$F_1$ score', fontsize=ACHSEN_FONTSIZE)
    axes[2].set_ylabel('')
    axes[2].tick_params(axis='y', labelsize=ACHSEN_FONTSIZE)
    axes[2].tick_params(axis='x', labelsize=REST_FONTSIZE)

    letters = ['a', 'b', 'c']
    for i, ax in axes.items():
        ax.yaxis.set_label_coords(-0.2, 0.5)  # Adjust the position of y-axis label
        # Add label in the bottom right corner
        x, y = -0.04, -0.135
        if i == 2:
            x, y = -.018, -0.13
        ax.text(x, y, f'{letters[i]})', transform=ax.transAxes,
                ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'),
            fontsize=REST_FONTSIZE)


    return fig, axes, df_scatter

def sc_phodi_vs_vicinity_model(mimir_phodi_vs_vicinity_dir: str) -> Tuple[plt.figure, List[plt.axes], List]:
    results, failed_measurements = get_mimir_result(mimir_phodi_vs_vicinity_dir)
    df = pd.DataFrame(results)
    results = [{**r, 'feature_generators': format_feature_generators(r.get('feature_generators'))} for r in results]
    df = pd.DataFrame(results)
    df['feature_generators'] = df.feature_generators.apply(phodi_vs_vicinity_format_feature_generators)
    df = df.groupby(['dataset', 'feature_generators', 'error_fraction', 'error_class']).agg(
                                                                                        {'f1': 'mean',
                                                                                        'precision': 'mean',
                                                                                        'recall': 'mean',
                                                                                        'run': list}).reset_index()
    df['normalized_dataset'] = df.apply(normalize_dataset, axis=1)

    df_pivot = pd.pivot(df, columns='feature_generators', index='normalized_dataset', values=['f1', 'precision', 'recall'])

    fig = plt.figure(constrained_layout=True)
    axes = fig.subplot_mosaic([[0, 1],[2, 2]])
    palette = sns.color_palette('Set3')

    # Remove spines
    sns.despine()

    for i, measure, label in [(0, 'precision', 'Precision'), (1, 'recall', 'Recall'),]:
        sns.scatterplot(x=df_pivot[measure]['Vicinity Model'], y=df_pivot[measure]['SC_Phodi'], ax=axes[i], color='black')
        sns.lineplot(x=[0, 1], y=[0, 1], ax=axes[i], color='black', alpha=.5)
        
        axes[i].set_xlabel(f'{label} Vicinity Model', fontsize=ACHSEN_FONTSIZE)
        axes[i].set_ylabel(f'{label} SC_Phodi', fontsize=ACHSEN_FONTSIZE)
        
        axes[i].set_xticks([0, 0.25, 0.5, 0.75, 1.00])
        axes[i].set_yticks([0, 0.25, 0.5, 0.75, 1.00])
        axes[i].set_aspect('equal', adjustable='box')

        axes[i].set_xticklabels([0, '', '', '', 1.00], fontsize=REST_FONTSIZE)
        axes[i].set_yticklabels([0, '', '', '', 1.00], fontsize=REST_FONTSIZE)
        

    sns.boxplot(y='feature_generators', x='f1', data=df, ax=axes[2], palette=MIMIR_LABEL_COLORS)
    # sns.swarmplot(data=df, x='f1', y='feature_generators', color="black", size=3.5, dodge=True, legend=False, ax=axes[2])

    axes[2].set_xlabel('$F_1$ score', fontsize=ACHSEN_FONTSIZE)
    axes[2].set_ylabel('')

    axes[2].tick_params(axis='y', labelsize=REST_FONTSIZE)
    axes[2].tick_params(axis='x', labelsize=REST_FONTSIZE)

    letters = ['a', 'b', 'c']
    for i, ax in axes.items():
        ax.yaxis.set_label_coords(-0.2, 0.5)  # Adjust the position of y-axis label
        # Add label in the bottom right corner
        x, y = -0.04, -0.135
        if i == 2:
            x, y = -.018, -0.13
        ax.text(x, y, f'{letters[i]})', transform=ax.transAxes,
                ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'),
            fontsize=REST_FONTSIZE)

    return fig, axes, failed_measurements