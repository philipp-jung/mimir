from pathlib import Path, PosixPath
import pandas as pd
import json
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from typing import List, Tuple, Dict
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

BARAN_DATASETS = ['beers', 'flights', 'food', 'hospital', 'rayyan', 'tax']
OPENML_DATASETS = ['6', '137', '151', '184', '1481', '41027', '43572']
RENUVER_DATASETS = ['bridges', 'cars', 'glass', 'restaurant']
RENUVER_NORMALIZED_DATASETS = ['bridges 1%', 'bridges 3%', 'cars 1%', 'cars 3%', 'glass 1%', 'glass 3%', 'restaurant 1%', 'restaurant 3%']

PATTERN_PALETTE = ['|', '', '//', '++', '\\\\\\', 'OO']

ABLATION_BAR_CUSTOM_LABELS = {
    'all models': 'All Correctors',
    'llm_master': '(no) RD_ImpFM',
    'llm_correction': '(no) ET_CorrFM',
    'fd': '(no) SC_Phodi',
    'auto_instance': '(no) IM_DataWig',
}
ABLATION_CORRECTOR_LABELS = {
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


ACHSEN_FONTSIZE = 9
REST_FONTSIZE = ACHSEN_FONTSIZE - 1

ABLATION_COLUMN_ORDER_1 = list(reversed(['all models', 'no llm_master', 'no llm_correction',  'no fd', 'no auto_instance',]))
ABLATION_COLUMN_ORDER_2 = list(reversed(['all models', 'llm_master', 'llm_correction',  'fd', 'auto_instance',]))

ABLATION_COLOR_PALETTE = sns.color_palette("Set3", 6)

MIMIR_LABEL_COLORS = {
    'no auto_instance': '#cae3e0',
    'no fd': '#dedde8',
    'no llm_correction': '#f9c6c0',
    'no llm_master': '#c3d6e1',
    'All Correctors': ABLATION_COLOR_PALETTE[5],
    'all models': ABLATION_COLOR_PALETTE[5],
    'Mimir': ABLATION_COLOR_PALETTE[5],
    'auto_instance': ABLATION_COLOR_PALETTE[0],
    'fd': ABLATION_COLOR_PALETTE[2],
    'llm_correction': ABLATION_COLOR_PALETTE[3],
    'llm_master': ABLATION_COLOR_PALETTE[4],
    'IM_DataWig': ABLATION_COLOR_PALETTE[0],
    'no IM_DataWig': '#cae3e0',
    'RD_ImpFM': ABLATION_COLOR_PALETTE[4],
    'no RD_ImpFM': '#c3d6e1',
    'SC_Phodi': ABLATION_COLOR_PALETTE[2],
    'no SC_Phodi': '#dedde8',
    'ET_CorrFM': ABLATION_COLOR_PALETTE[3],
    'no ET_CorrFM': '#f9c6c0',
    #'f1_et_corr': ABLATION_COLOR_PALETTE[3],
    'value_model': '#999999', # baran value models
    'Ensembled Value Model': '#999999',
    'Ensembled SC_Phodi': ABLATION_COLOR_PALETTE[2],
    'Ensembled Vicinity Model': '#999999',
    'Ensembled llm_correction': ABLATION_COLOR_PALETTE[3],
    'Vicinity Model': '#999999',
    'Baran': '#999999',
}

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

COLUMNWIDTH_PT = 238.96417
FIGURE_HEIGHT, FIGURE_WIDTH = set_size(COLUMNWIDTH_PT)


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
    if dataset in BARAN_DATASETS:
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
    elif row['dataset'] in BARAN_DATASETS or row['dataset'] in OPENML_DATASETS:
        return row['dataset']
    elif row['dataset'] in RENUVER_DATASETS:
        return f"{row['dataset']} {int(row['error_fraction'])}%"
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*2, FIGURE_HEIGHT*2), sharey=True)

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
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(2*FIGURE_WIDTH, 2*FIGURE_HEIGHT))

    # Plotting other models
    for col in pivot_df.columns:
        if col != 'all models':
            sns.scatterplot(x=pivot_df[col], y=pivot_df['all models'], label=col, marker='.', ax=ax)

    # Plotting the diagonal line
    ax.plot([-.2, 1.2], [-.2, 1.2], '-k', linewidth=.8)
    ax.set_aspect('equal', adjustable='box')

    # Adjust legend position
    handles, _ = ax.get_legend_handles_labels()
    labels = [ABLATION_CORRECTOR_LABELS[c] for c in pivot_df.columns if c != 'all models']
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=True, title='Ablations')

    # Set axis labels
    plt.xlabel("$F_1$ score Ablations")
    plt.ylabel("$F_1$ score Mimir's Ensemble")

    ax.xaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlim(-.05,1.05)
    ax.set_ylim(-.05,1.05)

    return fig, ax, failed_measurements
    

def plot_ablation(ablation_study_dir: str, corrector: str, subset=True):
    df_global, failed_experiments = get_ablation_study(ablation_study_dir)
    df_global = df_global.groupby(['dataset', 'feature_generators', 'error_fraction', 'error_class']).agg({'f1': 'mean',
                                                                                        'precision': 'mean',
                                                                                        'recall': 'mean',
                                                                                        'run': list}).reset_index()
    df_global['normal_dataset'] = df_global.apply(normalize_dataset, axis=1)
    
    if subset:  # filter out renuver 3% datasets because they do not contribute new insights to ablation
        df_global = df_global[~df_global['normal_dataset'].str.contains('3%')]
    
    pivot_df = df_global.pivot(index='normal_dataset', columns='feature_generators', values='f1')
    pivot_df.columns = [ABLATION_CORRECTOR_LABELS.get(x) for x in pivot_df.columns]
    
    df = pivot_df.loc[:, ['All Correctors', corrector, f'no {corrector}']]

    DATASETS_ORDER = list(reversed(OPENML_DATASETS)) + list(reversed(RENUVER_NORMALIZED_DATASETS)) + list(reversed(BARAN_DATASETS))
    DATASETS_ORDER = [x for x in DATASETS_ORDER if not '3%' in x]
    # Reindex the DataFrame according to the sorted index
    df = df.reindex(index=DATASETS_ORDER)

    # Create a figure and two sub-plots
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT*1.5))

    df.plot(kind='barh', ax=ax, rot=0, color=[MIMIR_LABEL_COLORS[label] for label in df.columns], legend=False, width=.75)
    ax.set_ylabel('Dataset', labelpad=0, fontsize=ACHSEN_FONTSIZE)
    ax.set_xlabel('$F_1$ score', labelpad=-8, fontsize=ACHSEN_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks([0.0, 1.0])  # Set x-axis ticks

    # Remove ticks from the y-axis
    ax.tick_params(axis='y', which='both', left=False, labelleft=True, pad=0, labelsize=REST_FONTSIZE)

    
    # Set custom y-axis labels with colors
    y_labels = df.index.to_list()

    datasets_worse = [d for d in y_labels if (df.loc[d, 'All Correctors'] + 0.05) < df.loc[d, f'no {corrector}']]
    datasets_better = [d for d in y_labels if df.loc[d, 'All Correctors'] > (df.loc[d, f'no {corrector}'] + 0.05)]
    y_colors = ['red' if dataset in datasets_worse else 'black' for dataset in y_labels]
    y_colors = ['green' if dataset in datasets_better else c for c, dataset in zip(y_colors, y_labels)]
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, rotation=0)

    for tick_label, color in zip(ax.get_yticklabels(), y_colors):
        tick_label.set_color(color)

    annotations = {d: {} for d in (datasets_better + datasets_worse)}
    for group_index, c in enumerate(ax.containers):
        for i, bar in enumerate(c):
            # Get the y-value corresponding to the current bar
            label = ax.get_yticklabels()[i].get_text()
            
            # Check if the label is in relevant_datasets
            if annotations.get(label) is not None:
                # Add label to the specific bar
                annotations[label][group_index] = {
                    'text': f'{bar.get_width():.2f}', 
                    'xy': (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    'xytext': (1, 0)}
    
    if corrector == 'SC_Phodi':
        annotations['flights'][2]['xytext'] = (1,1)
        annotations['flights'][0]['xytext'] = (1,-1)
    
    if corrector == 'ET_CorrFM':
        annotations['beers'][2]['xytext'] = (1,1)
        annotations['beers'][1]['xytext'] = (1,1)
        annotations['beers'][0]['xytext'] = (1,-2)

        annotations['rayyan'][2]['xytext'] = (1,1)
        annotations['rayyan'][1]['xytext'] = (1,1)
        annotations['rayyan'][0]['xytext'] = (1,-2)

        annotations['tax'][2]['xytext'] = (1,1)
        annotations['tax'][1]['xytext'] = (1,1)
        annotations['tax'][0]['xytext'] = (1,-2)

        annotations['cars 1%'][2]['xytext'] = (1,1)
        annotations['cars 1%'][1]['xytext'] = (1,1)
        annotations['cars 1%'][0]['xytext'] = (1,-2)
        
        annotations['glass 1%'][2]['xytext'] = (1,1)
        annotations['glass 1%'][1]['xytext'] = (1,1)
        annotations['glass 1%'][0]['xytext'] = (1,-2)

    if corrector == 'IM_DataWig':
        annotations['flights'][2]['xytext'] = (1,1)
        annotations['flights'][0]['xytext'] = (1,-1)

        annotations['food'][2]['xytext'] = (1,1)
        annotations['food'][0]['xytext'] = (1,-1)

        annotations['6'][2]['xytext'] = (1,1)
        annotations['6'][1]['xytext'] = (1,1)
        annotations['6'][0]['xytext'] = (1,-2)
 
        annotations['137'][2]['xytext'] = (1,1)
        annotations['137'][1]['xytext'] = (1,1)
        annotations['137'][0]['xytext'] = (1,-2)
 
        annotations['151'][2]['xytext'] = (1,1)
        annotations['151'][1]['xytext'] = (1,1)
        annotations['151'][0]['xytext'] = (1,-2)
 
        annotations['41027'][2]['xytext'] = (1,1)
        annotations['41027'][1]['xytext'] = (1,1)
        annotations['41027'][0]['xytext'] = (1,-1)
        
    if corrector == 'RD_ImpFM':
        annotations['43572'][2]['xytext'] = (1,1)
        annotations['43572'][1]['xytext'] = (1,1)
        annotations['43572'][0]['xytext'] = (1,-2)

    for dataset in annotations:
        for group_index in annotations[dataset]:
            a = annotations[dataset][group_index]
            ax.annotate(a['text'],
                        xy=a['xy'],
                        xytext=a['xytext'],
                        textcoords="offset points",
                        ha='left', va='center',
                        fontsize=REST_FONTSIZE)
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))
    ax.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.39, -0.12), fontsize=REST_FONTSIZE-2)

    return fig, ax, df


def plot_joint_ablation(ablation_study_dir: str, baranpp_global_performance_dir: str, mimir_global_performance_dir: str) -> Tuple[plt.figure, Tuple[plt.axes, plt.axes], List]:
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
    df.loc[df['normal_dataset'] == '151 \n cat 5%', 'normal_dataset'] = '151'
    df.loc[df['normal_dataset'] == '43572 \n cat 5%', 'normal_dataset'] = '43572'

    pivot_df = df.pivot(index='normal_dataset', columns='feature_generators', values='f1')

    # Reorder the columns of the pivot_df dataframe for the first sub-plot
    sub_pivot_df1 = pivot_df[ABLATION_COLUMN_ORDER_1]

    fig = plt.figure(constrained_layout=True)
    gs0 = gridspec.GridSpec(2, 2, figure=fig)

    gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[2:])
    ax0 = fig.add_subplot(gs00[:, 0])
    ax1 = fig.add_subplot(gs00[:, 1])

    gs10 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0:2])
    ax2 = fig.add_subplot(gs10[:, 0])
    ax3 = fig.add_subplot(gs10[:, 1])

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

    # Top left mimir vs baran++
    res_baranpp, failed_measurements = get_mimir_result(baranpp_global_performance_dir)
    res_baranpp = [{**r, 'ensemble': 'Baran++'} for r in res_baranpp]
    res_mimir, failed_measurements = get_mimir_result(mimir_global_performance_dir)
    df_baranpp = pd.DataFrame(res_baranpp + res_mimir)

    df_baranpp = (pd.pivot((df_baranpp.loc[:, ['normalized_dataset', 'ensemble', 'f1']]
            .groupby(['normalized_dataset', 'ensemble'])
            .agg({'f1': 'mean'})
            .reset_index()),
            columns='ensemble',
            index=['normalized_dataset'],
            values='f1')
            .round(2)
           )

    sns.scatterplot(x=df_baranpp['Baran++'], y=df_baranpp['Mimir'], color='000000', marker='o', ax=ax2, legend=False)

    # Plotting the diagonal line
    ax2.plot([-.2, 1.2], [-.2, 1.2], '-k', linewidth=.8)
    ax2.set_aspect('equal', adjustable='box')
    ax2.xaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
    ax2.yaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xlim(-.05,1.05)
    ax2.set_ylim(-.05,1.05)
    ax2.set_xlabel("$F_1$ score Baran++", fontsize=ACHSEN_FONTSIZE)
    ax2.set_ylabel("$F_1$ score Mimir", fontsize=ACHSEN_FONTSIZE)

    # plot top right ablation scatters
    pivot_df = df_scatter.pivot(index='normal_dataset', columns='feature_generators', values='f1')

    for col in pivot_df.columns:
        if col != 'all models':
            marker = 'X' if col.startswith('no') else 'o'
            sns.scatterplot(x=pivot_df[col], y=pivot_df['all models'], color=MIMIR_LABEL_COLORS[col], label=col, marker=marker, ax=ax3, legend=False)

    # Plotting the diagonal line
    ax3.plot([-.2, 1.2], [-.2, 1.2], '-k', linewidth=.8)
    ax3.set_aspect('equal', adjustable='box')
    ax3.xaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
    ax3.yaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xlim(-.05,1.05)
    ax3.set_ylim(-.05,1.05)


    h_select, l_select = ax1.get_legend_handles_labels()
    l_select = [ABLATION_BAR_CUSTOM_LABELS[c] for c in l_select]

    handles = [*reversed(h_select)]
    labels = [*reversed(l_select)]

    leg = fig.legend(
               handles,
               labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.18),
               ncol=3,
               title='Ablations',
               title_fontsize=ACHSEN_FONTSIZE,
               fontsize=ACHSEN_FONTSIZE)

    # Set axis labels
    ax3.set_xlabel("$F_1$ score Ablations", fontsize=ACHSEN_FONTSIZE)
    ax3.set_ylabel("$F_1$ score Mimir", fontsize=ACHSEN_FONTSIZE)

    letters = ['c', 'd', 'a', 'b',]
    for i, ax in enumerate([ax0, ax1, ax2, ax3]):
        ax.yaxis.set_label_coords(-0.2, 0.5)  # Adjust the position of y-axis label
        # Add label in the bottom right corner
        x, y = -0.04, -0.135
        if i == 2:
            x, y = -.018, -0.13
        ax.text(x, y, f'{letters[i]})', transform=ax.transAxes,
                ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'),
            fontsize=REST_FONTSIZE)

    return fig, (ax0, ax1, ax3), failed_measurements, pivot_df

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
            bbox_to_anchor=(0.5, -0.25, 0, 0), ncol=len(ABLATION_BAR_CUSTOM_LABELS)//2, frameon=False)

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
    
    res_mimir = [{**m, 'normalized_dataset': normalize_dataset(m)} for m in res_mimir]

    # Filter out 1pct OpenML because they're always perfectly cleaned
    res_mimir = [m for m in res_mimir if not (m['dataset_group'] == 'OpenML' and m['error_fraction'] == 1)]
    # Filter out non-categorical OpenML
    res_mimir = [r for r in res_mimir if not (r['dataset_group'] == 'OpenML' and 'simple_mcar' == r['error_class'])]

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
    hc_results = [{'normalized_dataset': normalize_dataset(m), 'dataset_group': get_dataset_group(m['dataset']), **m} for m in hc_results]

    # filter out 1% on OpenML because these are always perfectly cleaned
    hc_results = [r for r in hc_results if not (r['dataset_group'] == 'OpenML' and str(r['error_fraction']) == '1')]

    # Filter out 2%, 4%, 5% on RENUVER because other systems don't measure it.
    hc_results = [r for r in hc_results if not (r['dataset_group'] == "Renuver" and str(r['error_fraction']) in ['2', '4', '5'])]
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
                'precision': r.get('ec_p'),
                'recall': r.get('ec_r'),
                'f1': r.get('ec_f'),
                'error_class': r.get('error_class'),
                'dataset_group': get_dataset_group(r['dataset']),
                'error_fraction': r.get('error_fraction')} for r in results] 

    # filter out 1% on OpenML because these are always perfectly cleaned
    # and 10% on OpenML because others don't measure it.
    results = [r for r in results if not (r['dataset_group'] == 'OpenML' and str(r['error_fraction']) in ['1', '10'])]
    
    # Filter out 2%, 4%, 5% on RENUVER because others don't measure it.
    results = [r for r in results if not (r['dataset_group'] == 'Renuver' and str(r['error_fraction']) in ['2', '4', '5'])]
    return results

def plot_mimir_vs_baran(res_baran: list[dict], res_mimir: list[dict]) -> Tuple[plt.figure, plt.axes, pd.DataFrame, List]:
    """
    Boxplot global performance Mimir vs Baran.
    """
    df = pd.DataFrame([*res_mimir, *res_baran])

    df_melt = pd.melt(df, id_vars=['dataset_group', 'ensemble'], value_vars=['f1'], 
                  var_name='F1 Type', value_name='F1 Score')

    fig, ax = plt.subplots(figsize=(FIGURE_HEIGHT, FIGURE_WIDTH))

    
    # Create the boxplot
    sns.boxplot(data=df_melt, x='dataset_group', y='F1 Score', hue='ensemble', 
                palette=MIMIR_LABEL_COLORS, fliersize=2, linewidth=0.8, showfliers=False, 
                ax=ax, gap=.1)

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
            fontsize=8, 
            ncol=len(df_melt['ensemble'].unique())
            )

    # Limit the y-axis from 0 to 1
    ax.set_ylim([0, 1.05])

    return fig, ax, df_melt

def performance_table(mimir_results_path, baran_results_path, baranpp_results_path, garf_results_path, holoclean_results_path, full_metrics=True) -> Tuple[pd.DataFrame, List]:
    DATASETS_ORDER = BARAN_DATASETS + RENUVER_DATASETS + OPENML_DATASETS
    
    res_mimir, failed_measurements = get_mimir_result(mimir_results_path)
    res_mimir = [{**r, 'ensemble': 'Mimir'} for r in res_mimir]

    res_baran, failed_measurements = get_mimir_result(baran_results_path)
    res_baran = [{**r, 'ensemble': 'Baran'} for r in res_baran]

    res_baranpp, failed_measurements = get_mimir_result(baranpp_results_path)
    res_baranpp = [{**r, 'ensemble': 'Baran++'} for r in res_baranpp]

    res_hc = get_holoclean_global_performance(holoclean_results_path)
    res_hc = [{'normalized_dataset': r['normalized_dataset'],
                'ensemble': r['ensemble'],
                'dataset_group': r['dataset_group'],
                'error_class': r['error_class'],
                'f1': r['f1'],
                'precision': r['precision'],
                'recall': r['recall']} for r in res_hc]
    res_hc = [r for r in res_hc if not (r['dataset_group'] == 'OpenML' and 'simple_mcar' == r['error_class'])]

    res_garf = get_garf_global_performance(garf_results_path)
    res_garf = [{'normalized_dataset': r['normalized_dataset'],
                'ensemble': r['ensemble'],
                'dataset_group': r['dataset_group'],
                'error_class': r['error_class'],
                'f1': r['f1'],
                'precision': r['precision'],
                'recall': r['recall']} for r in res_garf]
    res_garf = [r for r in res_garf if not (r['dataset_group'] == 'OpenML' and 'simple_mcar' == r['error_class'])]

    df = pd.DataFrame([*res_mimir, *res_baranpp, *res_baran, *res_hc, *res_garf])
    
    if full_metrics:
        df_results = (pd.pivot((df.loc[:, ['normalized_dataset', 'ensemble', 'f1', 'precision', 'recall']]
                .groupby(['normalized_dataset', 'ensemble'])
                .agg({'f1': 'mean', 'precision': 'mean', 'recall': 'mean'})
                .reset_index()
                ),
                columns='ensemble',
                index='normalized_dataset',
                values=['f1', 'precision', 'recall'])
                .round(2)
                )
        df_results.columns = df_results.columns.swaplevel(0,1)  # group by correction system
        df_results.sort_index(inplace=True, axis=1)  # sort columns

    else:
        df_results = (pd.pivot((df.loc[:, ['normalized_dataset', 'ensemble', 'f1']]
                    .groupby(['normalized_dataset', 'ensemble'])
                    .agg({'f1': 'mean'})
                    .reset_index()
                    ),
                    columns='ensemble',
                    index='normalized_dataset',
                    values='f1')
                    .round(2)
                )
    
    df_results.sort_index(inplace=True, 
                          key=lambda index: [DATASETS_ORDER.index(x[0]) for x in index.str.split(' ')],
                          axis=0
                         )

    df_results = df_results.loc[:, ['Mimir', 'Baran++', 'Baran', 'HoloClean', 'Garf']]  # order columns
    return df_results


def et_corrfm_vs_value_model(mimir_result_dir: str) -> Tuple[plt.figure, List[plt.axes], pd.DataFrame]:
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
    dataset_subset = ['beers', 'flights', 'hospital', 'rayyan', 'cars 1%', 'cars 3%', 'glass 1%', 'glass 3%', 'tax', 'food']
    df_scatter = df.pivot(values=['f1', 'precision', 'recall'], index=['normalized_dataset'], columns=['feature_generators'])
    df_scatter_subset = df_scatter[df_scatter.index.isin(dataset_subset)]

    df_bars = df[df['normalized_dataset'].isin(dataset_subset)]
    df_bars = df_bars[df_bars['feature_generators'].isin(['llm_correction', 'value_model'])]

    fig = plt.figure(figsize=(2*FIGURE_HEIGHT, 2*FIGURE_WIDTH), constrained_layout=True)
    axes = fig.subplot_mosaic([['.', 0, 0, 1, 1], ['.', 0, 0, 1, 1], ['.', 2, 2, 2, 2]])
    palette = sns.color_palette('Set3')

    # Remove spines
    #sns.despine()

    annotations = {'precision': {}, 'recall': {}}
    for i, measure, label in [(0, 'precision', 'Precision'), (1, 'recall', 'Recall')]:
        sns.scatterplot(x=df_scatter[measure]['value_model'], y=df_scatter[measure]['llm_correction'], ax=axes[i], color=palette[4])
        sns.lineplot(x=[-.2, 1.2], y=[-.2, 1.2], ax=axes[i], color='black', linewidth=.8)
        
        axes[i].set_xlabel(f'{label} Value Model', fontsize=ACHSEN_FONTSIZE, labelpad=-9)
        axes[i].set_ylabel(f'{label} ET_CorrFM', fontsize=ACHSEN_FONTSIZE, labelpad=-6)

        axes[i].xaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
        axes[i].yaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xlim(-.05,1.05)
        axes[i].set_ylim(-.05,1.05)

        axes[i].set_aspect('equal', adjustable='box')

        # Plot highlighted datasets
        axes[i].scatter(x=df_scatter_subset[measure]['value_model'], y=df_scatter_subset[measure]['llm_correction'], color='black')

        # Annotate the scatter plot
        for dataset in dataset_subset:
            x_value, y_value = df_scatter[df_scatter.index == dataset][measure]['value_model'], df_scatter[df_scatter.index == dataset][measure]['llm_correction']
            annotations[measure][dataset] = {'type': 'annotate', 'coordinates': (x_value, y_value), 'xytext': (0, 10)}
            
    # Manually define annotations that look bad
    annotations['precision']['tax']['xytext'] = (-3, 9)
    annotations['precision']['food']['xytext'] = (19, 10)
    annotations['precision']['hospital']['xytext'] = (-2, -4)
    annotations['precision']['cars 3%']['xytext'] = (0, 9)

    annotations['precision']['flights']['type'] = 'arrow'
    annotations['precision']['flights']['xytext'] = (0, 20)

    annotations['precision']['rayyan']['type'] = 'arrow'
    annotations['precision']['rayyan']['xytext'] = (-20, -25)

    annotations['precision']['cars 1%']['type'] = 'arrow'
    annotations['precision']['cars 1%']['xytext'] = (-5, -25)

    annotations['precision']['beers']['type'] = 'arrow'
    annotations['precision']['beers']['xytext'] = (18, -12)

    annotations['precision']['glass 1%']['type'] = 'arrow'
    annotations['precision']['glass 1%']['xytext'] = (13, -20)

    annotations['precision']['glass 3%']['type'] = 'arrow'
    annotations['precision']['glass 3%']['xytext'] = (-25, -20)

    annotations['recall']['food']['xytext'] = (20, -1)
    annotations['recall']['beers']['xytext'] = (21, 9)
    annotations['recall']['glass 1%']['xytext'] = (32, 0)
    annotations['recall']['glass 3%']['xytext'] = (-4, -2)
    annotations['recall']['hospital']['xytext'] = (-4, 0)
    annotations['recall']['rayyan']['xytext'] = (27, -1)
    annotations['recall']['tax']['xytext'] = (-3, 9)

    annotations['recall']['cars 3%']['type'] = 'arrow'
    annotations['recall']['cars 3%']['xytext'] = (25, -4)

    annotations['recall']['flights']['type'] = 'arrow'
    annotations['recall']['flights']['xytext'] = (0, 20)

        
    for i, measure in enumerate(annotations.keys()):
        for dataset in annotations[measure]:
            an = annotations[measure][dataset]
            if an['type'] == 'annotate':
                axes[i].annotate(dataset, an['coordinates'], textcoords="offset points", xytext=an['xytext'], ha="right", va="top", fontsize=REST_FONTSIZE)
            elif an['type'] == 'arrow':
                axes[i].annotate(dataset,
                                 an['coordinates'],
                                 textcoords="offset points",
                                 xytext=an['xytext'],
                                 arrowprops={'arrowstyle': '->'},
                                 fontsize=REST_FONTSIZE)
            else:
                raise ValueError('unknown annotation type')

    df['feature_generators'] = df['feature_generators'].replace('all models', 'Ensembled llm_correction')
    custom_order = ['llm_correction', 'value_model', 'Ensembled llm_correction', 'Ensembled Value Model']
    sns.boxplot(y='feature_generators', x='f1', data=df, ax=axes[2], palette=MIMIR_LABEL_COLORS, order=custom_order, width=.6)
    custom_labels = ['ET_CorrFM', 'Value Model', 'Ensembled ET_CorrFM', 'Ensembled Value Model']

    axes[2].set_yticklabels(custom_labels)

    axes[2].set_xlabel('$F_1$ score', fontsize=ACHSEN_FONTSIZE)
    axes[2].set_ylabel('')

    axes[2].tick_params(axis='y', labelsize=REST_FONTSIZE)  # make datasets not too large
    axes[2].tick_params(axis='x', labelsize=REST_FONTSIZE)
    axes[2].set_xticks([0, .2, .4, .6, .8, 1])
    axes[2].set_xlim(-.05, 1.05)

    letters = ['a', 'b', 'c']
    for i, ax in axes.items():
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
    custom_order = ['SC_Phodi', 'Vicinity Model', 'Ensembled SC_Phodi', 'Ensembled Vicinity Model']
    df = df.groupby(['dataset', 'feature_generators', 'error_fraction', 'error_class']).agg(
                                                                                        {'f1': 'mean',
                                                                                        'precision': 'mean',
                                                                                        'recall': 'mean',
                                                                                        'run': list}).reset_index()
    df['normalized_dataset'] = df.apply(normalize_dataset, axis=1)

    df_pivot = pd.pivot(df, columns='feature_generators', index='normalized_dataset', values=['f1', 'precision', 'recall'])

    fig = plt.figure(figsize=(2*FIGURE_HEIGHT, 2*FIGURE_WIDTH), constrained_layout=True)
    axes = fig.subplot_mosaic([['.', 0, 0, 1, 1], ['.', 0, 0, 1, 1], ['.', 2, 2, 2, 2]])

    annotations = {'precision': {}, 'recall': {}}

    for i, measure, label in [(0, 'precision', 'Precision'), (1, 'recall', 'Recall'),]:
        sns.scatterplot(x=df_pivot[measure]['Vicinity Model'], y=df_pivot[measure]['SC_Phodi'], ax=axes[i], color='black')
        sns.lineplot(x=[-.2, 1.2], y=[-.2, 1.2], ax=axes[i], color='black', linewidth=.8)
        
        axes[i].set_xlabel(f'{label} Vicinity Model', fontsize=ACHSEN_FONTSIZE, labelpad=-9)
        axes[i].set_ylabel(f'{label} SC_Phodi', fontsize=ACHSEN_FONTSIZE, labelpad=-6)

        axes[i].xaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
        axes[i].yaxis.set_minor_locator(MultipleLocator(0.25))  # Set major ticks every 0.25
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xlim(-.05,1.05)
        axes[i].set_ylim(-.05,1.05)
        axes[i].tick_params(axis='y', labelsize=REST_FONTSIZE)
        axes[i].tick_params(axis='x', labelsize=REST_FONTSIZE)
        axes[i].set_aspect('equal', adjustable='box')

        x_value, y_value = df_pivot[df_pivot.index == 'flights'][measure]['Vicinity Model'], df_pivot[df_pivot.index == 'flights'][measure]['SC_Phodi']
        annotations[measure]['flights'] = {'type': 'arrow', 'coordinates': (x_value, y_value), 'xytext': (0, 10)}

    
    annotations['precision']['flights']['xytext'] = (-10, -20)
    annotations['recall']['flights']['xytext'] = (-35, 10)

    for i, measure in enumerate(annotations.keys()):
        for dataset in annotations[measure]:
            an = annotations[measure][dataset]
            if an['type'] == 'annotate':
                axes[i].annotate(dataset, an['coordinates'], textcoords="offset points", xytext=an['xytext'], ha="right", va="top", fontsize=REST_FONTSIZE)
            elif an['type'] == 'arrow':
                axes[i].annotate(dataset,
                                 an['coordinates'],
                                 textcoords="offset points",
                                 xytext=an['xytext'],
                                 arrowprops={'arrowstyle': '->'},
                                 fontsize=REST_FONTSIZE)
            else:
                raise ValueError('unknown annotation type')


    sns.boxplot(y='feature_generators', x='f1', data=df, ax=axes[2], palette=MIMIR_LABEL_COLORS, order=custom_order)

    axes[2].set_xlabel('$F_1$ score', fontsize=ACHSEN_FONTSIZE)
    axes[2].set_ylabel('')

    axes[2].tick_params(axis='y', labelsize=REST_FONTSIZE)  # make datasets not too large
    axes[2].tick_params(axis='x', labelsize=REST_FONTSIZE)
    axes[2].set_xticks([0, .2, .4, .6, .8, 1])
    axes[2].set_xlim(-.05, 1.05)

    letters = ['a', 'b', 'c']
    for i, ax in axes.items():
        # Add label in the bottom right corner
        x, y = -0.04, -0.135
        if i == 2:
            x, y = -.018, -0.13
        ax.text(x, y, f'{letters[i]})', transform=ax.transAxes,
                ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'),
            fontsize=REST_FONTSIZE)

    return fig, axes, failed_measurements

def plot_runtime(mimir_result_dir: str, baran_result_dir: str, baranpp_result_dir: str, correct_timeouts):
    res_mimir_runtime, failed_measurements = get_mimir_result(mimir_result_dir)
    res_baran_runtime, failed_measurements = get_mimir_result(baran_result_dir)
    res_baranpp_runtime, failed_measurements = get_mimir_result(baranpp_result_dir)
    
    res_baran_runtime = [r for r in res_baran_runtime if not (r['dataset_group'] == 'OpenML' and 'simple_mcar' == r['error_class'])]
    res_baranpp_runtime = [r for r in res_baranpp_runtime if not (r['dataset_group'] == 'OpenML' and 'simple_mcar' == r['error_class'])]

    DATASETS_ORDER = BARAN_DATASETS + RENUVER_DATASETS + OPENML_DATASETS
    res_mimir_runtime.sort(key=lambda x: DATASETS_ORDER.index(x['normalized_dataset'].split(' ')[0]), reverse=True)
    res_baran_runtime.sort(key=lambda x: DATASETS_ORDER.index(x['normalized_dataset'].split(' ')[0]), reverse=True)
    res_baranpp_runtime.sort(key=lambda x: DATASETS_ORDER.index(x['normalized_dataset'].split(' ')[0]), reverse=True)

    fig, ax = plt.subplots(figsize=(2*FIGURE_HEIGHT, 2*FIGURE_WIDTH))
    
    if correct_timeouts:
        timeout_datasets = [
            {'normalized_dataset': 'tax', 'runtime': 1E4},
            {'normalized_dataset': 'tax', 'runtime': 1E4},
            {'normalized_dataset': 'tax', 'runtime': 1E4},
            {'normalized_dataset': 'food', 'runtime': 1E4},
            {'normalized_dataset': 'food', 'runtime': 1E4},
            {'normalized_dataset': 'food', 'runtime': 1E4},
        ]
        res_baranpp_runtime = res_baranpp_runtime + timeout_datasets
    ax.scatter([r['runtime'] for r in res_mimir_runtime], [r['normalized_dataset'] for r in res_mimir_runtime], label='Mimir', marker='*')
    ax.scatter([r['runtime'] for r in res_baran_runtime], [r['normalized_dataset'] for r in res_baran_runtime], label='Baran', marker='.')
    ax.scatter([r['runtime'] for r in res_baranpp_runtime], [r['normalized_dataset'] for r in res_baranpp_runtime], label='Baran++', marker='+')


    ax.axvline(10000, color='#4e4e4d', linestyle='-', linewidth=.5)
    ax.annotate('Time  Limit', xy=(10000, 1), xytext=(5450, -0.81), horizontalalignment='left', color='#4e4e4d', fontsize=REST_FONTSIZE)
    
    plt.ylabel('Dataset', fontsize=ACHSEN_FONTSIZE)
    plt.xlabel('Runtime (s)', fontsize=ACHSEN_FONTSIZE)
    plt.xscale('log')  # Set x-axis scale to logarithmic
    plt.legend(fontsize=REST_FONTSIZE)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelsize=REST_FONTSIZE)
    #plt.grid(axis='x', linestyle=':')
    plt.grid(axis='y', linestyle=':')
    return fig, ax