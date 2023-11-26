from ipaddress import summarize_address_range
from re import T
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit 
import copy
import seaborn as sns
from ssNMR.formatting import format_plot
from lmfit import Parameters
from lmfit.models import ExponentialModel, GaussianModel, PseudoVoigtModel
import pprint
import pickle


def simple_parse_data(data_path):
    """
    DESCRIPTION: Parses a .txt file containing NMR spectra with frequency and intensity data and returns two arrays for each data column
    PARAMETERS: 
        data_path: path-like string
            the path to the raw .txt file
    RETURNS: [intensity, freq_Hz, freq_ppm, larmor_freq]
        intensity: numpy array
            a numpy array containing intensity data
        freq_Hz: numpy array
            a numpy array containing frequency data in Hz
        freq_ppm: numpy array
            a numpy array containing frequency data in freq_ppm
        larmor_freq: float
            the larmor frequency, calculated by dividing freq_Hz by freq_ppm
    """    
    # data starts on line 1, may need to change this!
    start_line = 1
    data = pd.read_csv(data_path, header=start_line, delimiter= ",")
    intensity = np.asarray([float(x) for x in data.iloc[:,1]])
    freq_Hz = np.asarray([float(x) for x in data.iloc[:,2]])
    freq_ppm = np.asarray([float(x) for x in data.iloc[:,3]])
    larmor_freq = freq_Hz[0]/freq_ppm[0]
    # print(freq_ppm)
    # print(intensity)
    return [intensity, freq_Hz, freq_ppm, larmor_freq]

def load_fit(data_file, fit_file, show_lgd=True, lgd_loc=0, lgd_fsize=22, 
        save_name=None, summary_save_dir=None, fig_save_dir=None, fit_save_dir=None, detect_fit_file=False,
        data_color='black', fit_color='red', init_fit_color='green', comp_colors=None, group_comp_colors=['blue', 'red'],
        comp_alphas=[0.5, 0.5], comp_fill_colors=['blue', 'red'], fill_curves=False, data_lw=1, plot_fit=True, ylim=None,
        element=None, nucleus=None):
    """
    DESCRIPTION: Replots a previous fit by reading a pickled data file.
    PARAMETERS: 
        data_file: path-like string
            The path to the raw .txt file containing intensity, frequency, and ppm NMR data
        fit_file: path-like string
        detect_fit_file: Boolean
        data_color: string
            Color to plot the NMR data with
        fit_color: string
            Color to plot the final fit with
        init_fit_color: string
            Color to plot the initial fit with
        comp_colors: string or array of strings
            List of colors to plot for each componenet, or a seaborn color pallete. If set to None, default colors are used
        group_comp_colors: array of strings, with length = len(comp_groups)
            Color to plot for each group of components
        show_lgd: boolean
            Whether or not to show the legend
        lgd_loc: int or string
            location of legend
        lgd_fsize: int
            font size of the legend
        fig_save_dir: string
            Directory to save figure into. If set to None, a figure will not be generated
        summary_save_dir: string
            Directory to save summary csv file to. If set to None, a summary will not be generated
        save_name: string
            The name to save the figure and summary as. If set to None, the data file prefix name will be used.
        element: string
            The element associated with the NMR spectra, e.g. 'Na'. Only used for plotting purposes
        nuclei: int
            The nuclei asssociated with the NMR spectra, e.g. 23. Only used for plotting purposes
    RETURNS: None
    """    
    if not save_name:
        save_name = os.path.splitext(os.path.basename(data_file))[0].replace('.txt', '')
    if detect_fit_file:
        file = open(fit_save_dir + save_name, 'rb')
    else:
        file = open(fit_file, 'rb')
    [fit_vals_dict, comps, fitx, fity, fit_ssb, ssb_list, fit_range, comp_names, comp_groups, group_names, ssb_comp_names] = pickle.load(file)
    # get NMR data
    intensity, freq_Hz, freq_ppm, larmor_freq = simple_parse_data(data_file)

    # generate summary of fits
    generate_summary(fit_vals_dict, comp_names, comp_groups, group_names, ssb_comp_names, summary_save_dir, save_name)
    # cut data to fit range
    lower_bound = next(x for x, val in enumerate(freq_ppm) if val <= fit_range[0])
    upper_bound = next(x for x, val in enumerate(freq_ppm) if val < fit_range[1])
    x = freq_ppm[lower_bound:upper_bound]
    y = intensity[lower_bound:upper_bound]
    # format plots
    if not isinstance(comp_colors, list) and comp_colors:
        sns.set_palette(comp_colors)
    # logic to deal with grouping components
    comp_group_index = []
    comp_labels = []
    for comp_name in comp_names:
        assigned_group = False
        for i, group in enumerate(comp_groups):
            if comp_name in group:
                comp_group_index.append(i)
                assigned_group = True
                if group_names[i] not in comp_labels:
                    comp_labels.append(group_names[i])
                else:
                    comp_labels.append(None)
        if not assigned_group:
            comp_group_index.append(-1)
            comp_labels.append(comp_name)
    # assigning colors to components
    colors = []
    default_colors = []
    ssb_colors = []
    fill_colors = []
    alphas = []
    for index in comp_group_index:
        if index != -1:
            colors.append(group_comp_colors[index])
            fill_colors.append(comp_fill_colors[index])
            alphas.append(comp_alphas[index])
        else:
            if not isinstance(comp_colors, list):
                color = next(ax._get_lines.prop_cycler)['color']
            else:
                color = comp_colors.pop()
            colors.append(color)
            default_colors.append(color)
    for i in range(len(ssb_list)):
        color_index = 0
        for index in comp_group_index:
            if index != -1:
                ssb_colors.append(group_comp_colors[index])
            else:
                ssb_colors.append(default_colors[color_index])
                color_index += 1
    plt, ax = format_plot(
        fig_size=(10,5),
        hide_y=True,
    )
    x= x[::-1]
    # plt.plot(x, y[::-1], color=data_color, label=None, linewidth=data_lw)
    if plot_fit:
        # plt.plot(fitx, fity, '-', label='fit', color=fit_color, linewidth=1)
        for i, comp_name in enumerate([comp_names[4], comp_names[1]]):
            # print(comp_name+'_')
            plt.plot(fitx, comps[comp_name+'_'][::-1], '--', label=comp_labels[i], color=colors[i])
            if fill_curves:
                ax.fill_between(fitx, comps[comp_name+'_'][::-1], color=fill_colors[i], alpha=alphas[i])
        if fit_ssb:
            for i, ssb_comp_name in enumerate(ssb_comp_names):
                plt.plot(fitx, comps[ssb_comp_name][::-1], '--', color=ssb_colors[i])
    if element and nucleus:
        nmr_nucleus = '$^{%d}$%s' % (nucleus, element)
    else:
        nmr_nucleus = ''
    plt.xlabel('\u03B4{} / ppm'.format(nmr_nucleus))
    plt.xlim(fit_range)
    if ylim:
        plt.ylim(ylim)
    if show_lgd:
        lgnd = plt.legend(loc=lgd_loc, labelspacing=0.2, fontsize=lgd_fsize, handlelength=1, frameon=False)
        # for line in lgnd.get_lines():
        #     line.set_linewidth(2)
        lgnd.set_draggable(True)
    plt.tight_layout()
    if fig_save_dir:
        plt.savefig(fig_save_dir + save_name + '-fit' + '.png')
    plt.show()
    plt.close()
    return

def generate_summary(model_result, comp_names, comp_groups, group_names, ssb_comp_names, save_dir, base_file_name):
    if not isinstance(model_result, dict):
        fit_vals_dict = model_result.best_values
    else:
        fit_vals_dict = model_result
    component_df = pd.DataFrame(columns=['component(s)', 'group', 'relative %', 'amplitude','center','sigma','fraction', 'fwhm','height'])
    # print('comp groups: ', comp_groups)
    new_comp_names = []
    comp_amplitudes = []
    comp_centers = []
    comp_sigmas = []
    comp_fractions = []
    comp_fwhms = []
    comp_heights = []
    comp_group_names = []
    for comp_name in comp_names:
        prefix = comp_name + '_'
        amplitude = fit_vals_dict['{}{}'.format(prefix, 'amplitude')]
        center = fit_vals_dict['{}{}'.format(prefix, 'center')]
        sigma = fit_vals_dict['{}{}'.format(prefix, 'sigma')]
        fraction = fit_vals_dict['{}{}'.format(prefix, 'fraction')]
        fwhm = 2*sigma
        height = (((1-fraction)*amplitude)/max(1e-15, (sigma*np.sqrt(np.pi/np.log(2))))+(fraction*amplitude)/max(1e-15, (np.pi*sigma)))
        new_comp_names.append(comp_name)
        comp_group_names.append('n/a')
        comp_amplitudes.append(amplitude)
        comp_centers.append(center)
        comp_sigmas.append(sigma)
        comp_fractions.append(fraction)
        comp_fwhms.append(fwhm)
        comp_heights.append(height)
    if len(ssb_comp_names) > 0:
        for ssb_comp_name in ssb_comp_names:
            amplitude = fit_vals_dict['{}{}'.format(ssb_comp_name, 'amplitude')]
            center = fit_vals_dict['{}{}'.format(ssb_comp_name, 'center')]
            sigma = fit_vals_dict['{}{}'.format(ssb_comp_name, 'sigma')]
            fraction = fit_vals_dict['{}{}'.format(ssb_comp_name, 'fraction')]
            fwhm = 2*sigma
            height = (((1-fraction)*amplitude)/max(1e-15, (sigma*np.sqrt(np.pi/np.log(2))))+(fraction*amplitude)/max(1e-15, (np.pi*sigma)))
            new_comp_names.append(ssb_comp_name.rstrip('_'))
            comp_group_names.append('n/a')
            comp_amplitudes.append(amplitude)
            comp_centers.append(center)
            comp_sigmas.append(sigma)
            comp_fractions.append(fraction)
            comp_fwhms.append(fwhm)
            comp_heights.append(height)
    comp_relative_amps = [100*x/sum(comp_amplitudes) for x in comp_amplitudes]
    comp_group_amplitudes = []
    original_comp_groups = comp_groups.copy()
    new_comp_groups = copy.deepcopy(comp_groups)
    original_comp_group_names = comp_group_names.copy()
    if len(ssb_comp_names) > 0:
        for ssb_comp_name in ssb_comp_names:
            for i, original_comp_group_names in enumerate(original_comp_groups):
                if any(name in ssb_comp_name for name in original_comp_group_names):
                    new_comp_groups[i].append(ssb_comp_name)
    # print('comp groups3: ', comp_groups)
    # print('comp_groups: ', comp_groups)
    # print('group names: ', group_names)
    # print('comp_group_names: ', comp_group_names)
    # print('comp names: ', new_comp_names)
    # print(len(comp_groups))
    groupless_comp_amps = comp_amplitudes.copy()
    # print('new comp groups: ', new_comp_groups)
    # print('group names: ', group_names)
    for group, name in zip(new_comp_groups, group_names):
        total_amplitude = 0
        iso_amplitudes = []
        centers = []
        comps = ''
        for comp in group:
            comps += '{} '.format(comp)
            # getting center data, just for isotropic peaks
            for i in range(len(new_comp_names)):
                if comp == new_comp_names[i]:
                    iso_amplitude = comp_amplitudes[i]
                    center = comp_centers[i]
                    iso_amplitudes.append(iso_amplitude)
                    centers.append(center)
            # getting amplitude data for all peaks
            for i in range(len(new_comp_names)):
                if comp.rstrip('_') == new_comp_names[i]:
                    amplitude = comp_amplitudes[i]
                    total_amplitude += amplitude
        # print('iso_amplitudes: ', iso_amplitudes)
        mean_center = np.average(centers, weights=iso_amplitudes)
        comp_sigmas.append('n/a')
        comp_fractions.append('n/a')
        comp_fwhms.append('n/a')
        comp_heights.append('n/a')
        comp_centers.append(mean_center)
        comp_group_amplitudes.append(total_amplitude)
        comp_group_names.append(name)
        new_comp_names.append(comps)
    # print('new comp names: ', new_comp_names)
    group_relative_amps = [100*x/sum(comp_amplitudes) for x in comp_group_amplitudes]
    [comp_amplitudes.append(x) for x in comp_group_amplitudes]
    [comp_relative_amps.append(x) for x in group_relative_amps]
    comp_data = [new_comp_names, comp_group_names, comp_relative_amps, comp_amplitudes, comp_centers, comp_sigmas, comp_fractions, 
    comp_fwhms, comp_heights]
    for name, data in zip(list(component_df.columns), comp_data):
        # print(data)
        component_df[name] = data
        # print('added {}'.format(name))
    if save_dir:
        component_df.to_csv(save_dir + base_file_name + '.csv')
    return [groupless_comp_amps, comp_group_amplitudes]

def get_components_list(fit_vals_dict, comp_names):
    components_list = []
    for i, comp_name in enumerate(comp_names):
        prefix = comp_name + '_'
        comp = {
            'amplitude': fit_vals_dict['{}{}'.format(prefix, 'amplitude')],
            'center': fit_vals_dict['{}{}'.format(prefix, 'center')],
            'sigma': fit_vals_dict['{}{}'.format(prefix, 'sigma')],
            'fraction': fit_vals_dict['{}{}'.format(prefix, 'fraction')],
        }
        components_list.append(comp)
    return components_list

def print_fit_vals(model_result, comp_names):
    fit_vals_dict = model_result.best_values
    for i, comp_name in enumerate(comp_names):
        prefix = comp_name + '_'
        comp = {
            'amplitude': fit_vals_dict['{}{}'.format(prefix, 'amplitude')],
            'center': fit_vals_dict['{}{}'.format(prefix, 'center')],
            'sigma': fit_vals_dict['{}{}'.format(prefix, 'sigma')],
            'fraction': fit_vals_dict['{}{}'.format(prefix, 'fraction')],
        }
        print('comp{}=\\'.format(i+1))
        pprint.pprint(comp)

def fit_T2(save_dir, save_name, L1_data, intensity_data, spin_rate, labels=None, normalize=False,
    show_plot=True, colors=['red', 'blue', 'green']):
    """
    DESCRIPTION:  Given rotor delay and intensity data for a T2 experiment, extract out the T2 time constant in ms
    PARAMETERS:  
        save_dir: string
            Directory to save plot figure
        save_name: string
            Figure save file name
        L1_data: array of arrays
            List of of rotor delay data each acquistion was run at, for each resonance, i.e. [L1_para, L1_dia],
            where L1_para = L2_para = [1,2,3,4,5,6,7,8,10,20,30,40,50,60,70,80]
        intensity_data: array of arrays
            List of the intensity values extracted from a component/ group of components after fitting the spectra
        spin_rate: integer
            The MAS spin rate of the rotor, in Hz
        labels: array of strings
            Label names for each component/ group of components
        normalize: boolean
            Whether or not to normalize the plot values.
    RETURNS: [T2_list, unscaled_percentages, scaled_percentages]
        T2_list: array of floats
            list of T2 constants corresponding to each component/ group of components specified in intensity_data, index-matched
        unscaled_percentages: array of floats
            list of unscaled molar percentages of each component/ group of components specified in intensity_data, index-matched
        scaled_percentages: array of floats
            list of T2 scaled molar percentages of each component/ group of components specified in intensity_data, index-matched
    """
    plt, ax = format_plot(
        fig_size=(8,8),
    )
    extracted_intensities = []
    initial_intensities = []
    label_list = []
    T2_list = []
    # print('L1_data: {}'.format(L1_data))
    # print('intensity_data: {}'.format(intensity_data))
    # print('labels: {}'.format(labels))
    norm_factor = [intensity[0] for intensity in intensity_data]
    for i in range(len(L1_data)):
        L1 = np.array(L1_data[i])
        intensity = np.array(intensity_data[i])
        if normalize:
            initial_intensities.append(intensity[0])
            intensity = intensity / intensity[0]
        else:
            initial_intensities.append(intensity[0])
        # converting L1 to delay time, in milliseconds
        time = 2/spin_rate*L1*1000
        popt, pcov = curve_fit(T2_decay_func, time, intensity, p0=[time[0], intensity[0]])
        T2 = popt[0]
        init_intensity = popt[1]
        std_dev = np.sqrt(np.diag(pcov))
        T2_std_dev = std_dev[0]
        init_intensity_std_dev = std_dev[1]
        if normalize:
            abs_init_intensity = init_intensity*norm_factor[i]
            abs_init_intensity_std_dev = init_intensity_std_dev*norm_factor[i]
        else:
            abs_init_intensity = init_intensity
            abs_init_intensity_std_dev = init_intensity_std_dev
        extracted_intensities.append(abs_init_intensity)
        T2_list.append(T2)
        if not labels:
            label = 'Feature {}'.format(i)
        else:
            label = labels[i]
        label_list.append(label)
        print('-----------------------------------------------')
        print('*****{} fitting results*****'.format(label))
        print('-----------------------------------------------')
        print('T2 constant: {} ms'.format(np.round(T2, 4)))
        print('T2 constant std dev: {}'.format(np.round(T2_std_dev, 4)))
        print('Initial intensity: {}'.format(np.round(abs_init_intensity, 0)))
        print('Initial intensity std dev: {}'.format(np.round(abs_init_intensity_std_dev, 0)))
        plt.plot(time, intensity, 'o', color=colors[i], label=label)
        plt.plot(time, T2_decay_func(time, T2, init_intensity), '-', color='black')
    print('-----------------------------------------------')
    print('*****scaled intensity results*****'.format(label))
    print('-----------------------------------------------')
    unscaled_percentages = [intensity/sum(initial_intensities)*100 for intensity in initial_intensities]
    scaled_percentages = [intensity/sum(extracted_intensities)*100 for intensity in extracted_intensities]*100
    for i in range(len(L1_data)):
        print('Unscaled {} quantification: {}%'.format(label_list[i], np.round(unscaled_percentages[i], 3)))
        print('T2 scaled {} quantification: {}%'.format(label_list[i], np.round(scaled_percentages[i], 3)))
    plt.xlabel('Time (ms)')
    if normalize:
        plt.ylabel('Normalized Intensity (a.u.)')
    else:
        plt.ylabel('Intensity (a.u.)')
    plt.legend(prop={'size': 22}, frameon=False).set_draggable(True)
    plt.savefig(save_dir + save_name + '.png')
    if show_plot:
        plt.show()
    plt.close()
    return [T2_list, unscaled_percentages, scaled_percentages]

def T2_decay_func(time, T2, init_intensity):
    #fit T2 and init_intensity
    return init_intensity*np.exp(-1*time/T2)

def fit_T2_spectra(data_files, rotor_periods, fit_range, components_list=None, comp_constraints=None, 
        comp_names=None, normalize=False, comp_groups=[], group_names=[],
        fit_ssb=False, ssb_list=[], mas_freq=60000,
        print_results=True, show_plot=True, plot_init_fit=True, show_lgd=True, lgd_loc=0, lgd_fsize=22, 
        save_name=None, summary_save_dir=None, fig_save_dir=None,
        data_color='black', fit_color='red', init_fit_color='green', comp_colors=None,   group_comp_colors=['blue', 'red']):
    """
    DESCRIPTION:  Given a set of T2 relaxation data, automatically fit all spectra, and extract of T2 constants and
                  scaled intensity values for all components
    PARAMETERS:  
        data_files: list of strings
            List of files containing T2 relaxation experiments, with varying interpulse delays
        rotor periods: array of integers
            List of rotor delays for each of the spectra in data_files, index-matched
        normalize: boolean
            Whether or not to normalize the plot for T2 intensity decay
        **kwargs: key-word arguments
            key-word arguments corresponding to the 'fit' function. See 'fit' function for details
    RETURNS: [T2_list, unscaled_percentages, scaled_percentages]
        T2_list: array of floats
            list of T2 constants (in ms) corresponding to each component/ group of components specified in intensity_data, index-matched
        unscaled_percentages: array of floats
            list of unscaled molar percentages of each component/ group of components specified in intensity_data, index-matched
        scaled_percentages: array of floats
            list of T2 scaled molar percentages of each component/ group of components specified in intensity_data, index-matched
    """
    amplitudes = []
    comp_group_index = []
    comp_labels = []
    plt, ax = format_plot(
        fig_size=(8,8),
    )

        
    for comp_name in comp_names:
        assigned_group = False
        for i, group in enumerate(comp_groups):
            if comp_name in group:
                comp_group_index.append(i)
                assigned_group = True
                if group_names[i] not in comp_labels:
                    comp_labels.append(group_names[i])
                else:
                    comp_labels.append(None)
        if not assigned_group:
            comp_group_index.append(-1)
            comp_labels.append(comp_name)
    # assigning colors to components
    colors = []
    default_colors = []
    for index in comp_group_index:
        if index != -1:
            colors.append(group_comp_colors[index])
        else:
            color = next(ax._get_lines.prop_cycler)['color']
            colors.append(color)
            default_colors.append(color)
    if len(comp_groups) > 0:
        for i in range(len(comp_groups)):
            amplitudes.append([])
    else:
        for i in range(len(components_list)):
            amplitudes.append([])
    if len(comp_groups) > 0:
        colors = group_comp_colors
    plt.close()
    for data_file in data_files:
        save_name = os.path.splitext(os.path.basename(data_file))[0].replace('.txt', '')
        freq_ppm_data, intensity_data, model_result, groupless_amplitudes, group_amplitudes= \
        fit(data_file=data_file, fit_range=fit_range, components_list=components_list, comp_constraints=comp_constraints,
        comp_names=comp_names, comp_groups=comp_groups, group_names=group_names, fit_ssb=fit_ssb, ssb_list=ssb_list,
        mas_freq=mas_freq, print_results=print_results, show_plot=show_plot, plot_init_fit=plot_init_fit,
        show_lgd=show_lgd, lgd_loc=lgd_loc, lgd_fsize=lgd_fsize, save_name=save_name, summary_save_dir=summary_save_dir,
        fig_save_dir=fig_save_dir, data_color=data_color, fit_color=fit_color, init_fit_color=init_fit_color,
        comp_colors=comp_colors, group_comp_colors=group_comp_colors)
        if len(comp_groups) > 0:
            for i in range(len(comp_groups)):
                amplitudes[i].append(group_amplitudes[i])
        else:
            for i in range(len(components_list)):
                amplitudes[i].append(groupless_amplitudes[i])
    if len(comp_groups) > 0:
        rotor_period_data = len(comp_groups)*[rotor_periods]
        labels = group_names
    else: 
        rotor_period_data = len(components_list)*[rotor_periods]
        labels = comp_names
    T2_list, unscaled_percentages, scaled_percentages = fit_T2(
        save_dir=fig_save_dir,
        save_name=save_name,
        L1_data=rotor_period_data,
        intensity_data=amplitudes,
        spin_rate=mas_freq,
        labels=labels,
        normalize=normalize,
        colors=colors,
        show_plot=True
    )
    return [T2_list, unscaled_percentages, scaled_percentages]

def fit(data_file, fit_range, components_list=None, comp_constraints=None, fit_file=None, comp_names=None, comp_groups=[], group_names=[],
        fit_ssb=False, ssb_list=[], mas_freq=60000,
        print_results=True, show_plot=True, plot_init_fit=True, show_lgd=True, lgd_loc=0, lgd_fsize=22, 
        save_name=None, summary_save_dir=None, fig_save_dir=None, fit_save_dir=None, element = None, nucleus = None,
        data_color='black', fit_color='red', init_fit_color='green', comp_colors=None, group_comp_colors=['blue', 'red'], 
        ):
    """
    DESCRIPTION: Given NMR frequency and intensity data and a model consisting of pseudo-voigt components, fits the NMR spectra
    PARAMETERS: 
        data_file: path-like string
            The path to the raw .txt file containing intensity, frequency, and ppm NMR data
        fit_range: array of floats
            the range to fit the NRM data over, in ppm
        components_list: array of dictionaries, i.e., [component0, component1, ...], where component0 is a dictionary object
            A list of psdeuo-voigt components to fit NMR data to. Each componenent is a dictionary in the following format:
                component0 = 
                    {'amplitude': 50000,
                    'center': 6,
                    'fraction': 1,
                    'sigma': 12}
            The dictionary keys must be identical to the keys above, and initial guesses must be provided. 
            amplitude: the integrated area under the component
            center: the center of the component, in ppm
            fraction: the ratio of lorenztian to gaussian for the component. 1 is a pure lorenztian, 0 is a pure gaussian
            sigma: related to the variance, or fwhm of the component
            for more details, refer to the following link: https://lmfit.github.io/lmfit-py/builtin_models.html#pseudovoigtmodel
        comp_constraints: array of dictionaries, i.e., [constraint0, constraint1, ...], where constraint0 is a dictionary object
            A list of constraints for each pseudo-voigt component, indexed-matched to 'components_list'. Each constraint is a 
            dictionary in the following format:
                comp0_constraints = {
                    'amplitude_vary' : True,
                    'amplitude_min' : None,
                    'amplitude_max' : None,
                    'amplitude_expr' : None,
                    'center_vary' : True,
                    'center_min' : None,
                    'center_max' : None,
                    'center_expr' : None,
                    'fraction_vary' : True,
                    'fraction_min' : None,
                    'fraction_max' : None,
                    'fraction_expr' : None,
                    'sigma_vary' : True,
                    'sigma_min' : None,
                    'sigma_max' : None,
                    'sigma_expr' : None,
                }
            The dictionary keys must be identical to the keys above. Not all constraints must be specified; default values
            (specified in the example above) will be applied. An empty dictionary can be used, but a constraint must always be 
            specified for a component. For each variable (amplitude, center, fraction, sigma), there are four constraints that
            can be applied. 'vary' indicates whether or not to optimize that variable. 'min' and 'max' provide bounds for the 
            variable. 'expr' provides a way to define mathematical constraints. For example, the relative ratio of amplitudes of
            two components (comp_0 and comp_1) can be set as such:
                    component0 = 
                    {'amplitude': 50000,
                    'center': 6,
                    'fraction': 1,
                    'sigma': 12}
                    component1 = 
                    {'amplitude': 50000,
                    'center': 6,
                    'fraction': 1,
                    'sigma': 12}
                    components_list = [component0, component1]
                    comp_names = ['p1', 'p2']
                    comp0_constraints = {
                        amplitude_expr' : '2*p1_amplitude',
                    }
                    comp1_constrains = {}
                    comp_constraints = [comp0_constraints, comp1_constraints]
        comp_names: array of strings
            Names assigned to each of the components, index-matched with components_list. Used in labels for plots, 
            and also as variable names to be used in 'expr' constraints (see above). Must not contain mathematical symbols
            such as '-, +, sin, etc...'
        comp_groups: nested array, i.e., [group1, group2], where group1 = [comp1, comp2, comp3] and group2 = [comp4, comp5]
            Optional. Grouping of several or all components defined in comp_names. Useful for associating several components which 
            describe the same environment together. Amplitude calculations and plot labels and colors will use these groupings.
            Each component must be a string that is listed in comp_names.
        group_names: array of strings, where len(group_names) = len(comp_names)
            Names to define for each grouping of components defined in comp_groups
        fit_ssb: boolean
            Whether or not to enable spinning side band fitting
        ssb_list: array of integers
            An list of spinning sideband indices to include in the fitting. [1, -1] indicates two spinning side bands, 
            one on either side of the isotropic peaks, spaced by 1*mas_freq
        mas_freq: integer
            The MAS spinning speed, in Hz.
        data_color: string
            Color to plot the NMR data with
        fit_color: string
            Color to plot the final fit with
        init_fit_color: string
            Color to plot the initial fit with
        comp_colors: string or array of strings
            List of colors to plot for each componenet, or a seaborn color pallete. If set to None, default colors are used
        group_comp_colors: array of strings, with length = len(comp_groups)
            Color to plot for each group of components
        show_plot: boolean
            Whether or not to show the plot
        plot_init_fit: boolean
            Whether or not to plot the initial fit
        show_lgd: boolean
            Whether or not to show the legend
        lgd_loc: int or string
            location of legend
        lgd_fsize: int
            font size of the legend
        figure_save_dir: string
            Directory to save figure into. If set to None, a figure will not be generated
        summary_save_dir: string
            Directory to save summary csv file to. If set to None, a summary will not be generated
        save_name: string
            The name to save the figure and summary as. If set to None, the data file prefix name will be used.
        element: string
            The element associated with the NMR spectra, e.g. 'Na'. Only used for plotting purposes
        nucleus: int
            The nuclei asssociated with the NMR spectra, e.g. 23. Only used for plotting purposes
    RETURNS: [freq_ppm_data, intensity_data, model_result]
        freq_ppm_data: numpy array
            a numpy array containing frequency data in freq_ppm
        intensity_data: numpy array
            a numpy array containing intensity data
        model_result: lmfit.model.ModelResult class
            results of the fit, contained in a ModelResult object: https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult
    """    
    # get NMR data
    intensity, freq_Hz, freq_ppm, larmor_freq = simple_parse_data(data_file)
    if not save_name:
        save_name = os.path.splitext(os.path.basename(data_file))[0].replace('.txt', '')
    if fit_file:
        file = open(fit_file, 'rb')
        [fit_vals_dict, comps_old, fitx_old, fity_old, fit_ssb_old, ssb_list_old, fit_range_old, 
        comp_names, comp_groups, group_names, ssb_comp_names_old] = pickle.load(file)
        components_list = get_components_list(fit_vals_dict, comp_names)
    # checking inputs
    if len(components_list) > len(comp_constraints):
        raise ValueError("Number of component constraints ({}) is less than the number of components ({})"\
            .format(len(comp_constraints), len(components_list)))
    # cut data to fit range
    lower_bound = next(x for x, val in enumerate(freq_ppm) if val <= fit_range[0])
    upper_bound = next(x for x, val in enumerate(freq_ppm) if val < fit_range[1])
    x = freq_ppm[lower_bound:upper_bound]
    y = intensity[lower_bound:upper_bound]
    # format plots
    if not isinstance(comp_colors, list) and comp_colors:
        sns.set_palette(comp_colors)
    plt, ax = format_plot(
        fig_size=(8,8),
        hide_y=True,
    )
    # generate isotropic pseudo-voigt components based on inputs
    compiled_components = []
    pars = Parameters()
    for i in range(len(components_list)):
        comp_name = comp_names[i]
        component = components_list[i]
        constraints = comp_constraints[i]
        prefix=comp_name+'_'
        pseudo_voigt = PseudoVoigtModel(prefix=prefix)
        pars.update(pseudo_voigt.make_params())
        params_list = ['fraction', 'center', 'sigma', 'amplitude']
        constraints_list = ['vary', 'min', 'max', 'expr']
        # set intitial values and constraints for each parameter for pseudo-voigt lineshape
        for param_name in params_list:
            param = pars['{}{}'.format(prefix, param_name)]
            # set initial value
            param.set(value=component[param_name])
            # set constraints
            for constraint_name in constraints_list:
                param_constraint_name = '{}_{}'.format(param_name, constraint_name)
                if param_constraint_name in constraints.keys():
                    constraint = constraints['{}_{}'.format(param_name, constraint_name)]
                    if constraint is not None:
                        if constraint_name == 'vary':
                            param.set(vary=constraint)
                        elif constraint_name == 'min':
                            param.set(min=constraint)
                        elif constraint_name == 'max':
                            param.set(max=constraint)
                        elif constraint_name == 'expr':
                            param.set(expr=constraint)
        compiled_components.append(pseudo_voigt)
    # ssb_list = [-2, 2, -1, 1]
    ssb_comp_names = []
    if fit_ssb:
        for index in ssb_list:
            for i in range(len(components_list)):
                comp_name = comp_names[i]
                component = components_list[i]
                constraints = comp_constraints[i]
                if index > 0:
                    index_name = index
                else:
                    index_name = 'neg{}'.format(str(np.abs(index)))
                prefix=comp_name+'_ssb_{}_'.format(index_name)
                ssb_comp_names.append(prefix)
                pseudo_voigt = PseudoVoigtModel(prefix=prefix)
                pars.update(pseudo_voigt.make_params())
                params_list = ['fraction', 'center', 'sigma', 'amplitude']
                constraints_list = ['vary', 'min', 'max', 'expr']
                # set intitial values and constraints for each parameter for pseudo-voigt lineshape
                for param_name in params_list:
                    param = pars['{}{}'.format(prefix, param_name)]
                    iso_param_name = '{}_{}'.format(comp_name, param_name)
                    if param_name == 'fraction' or param_name == 'sigma':
                        param.set(expr='{}'.format(iso_param_name))
                    elif param_name == 'center':
                        param.set(expr='{}+{}*{}'.format(iso_param_name, index, mas_freq/larmor_freq))
                    elif param_name == 'amplitude':
                        param.set(value=component[param_name], min=0)
                compiled_components.append(pseudo_voigt)
    # build model containing all generated components
    model = compiled_components[0]
    for i in range(1, len(compiled_components)):
        model = model + compiled_components[i]
    init = model.eval(pars, x=x)
    # fit components to data
    out = model.fit(y, pars, x=x)
    # print fitting results
    if print_results:
        print(out.fit_report(min_correl=0.5))
    comps = out.eval_components(x=x)
    # print(out.best_values)
    # print fitting results as dictionary objects
    if print_results:
        print_fit_vals(out, comp_names)
    # generate summary of fits
    groupless_amplitudes, group_amplitudes = generate_summary(out, comp_names, comp_groups, group_names, 
    ssb_comp_names, summary_save_dir, save_name)
    
    # logic to deal with grouping components
    comp_group_index = []
    comp_labels = []
    for comp_name in comp_names:
        assigned_group = False
        for i, group in enumerate(comp_groups):
            if comp_name in group:
                comp_group_index.append(i)
                assigned_group = True
                if group_names[i] not in comp_labels:
                    comp_labels.append(group_names[i])
                else:
                    comp_labels.append(None)
        if not assigned_group:
            comp_group_index.append(-1)
            comp_labels.append(comp_name)
    # assigning colors to components
    colors = []
    default_colors = []
    ssb_colors = []
    for index in comp_group_index:
        if index != -1:
            colors.append(group_comp_colors[index])
        else:
            if not isinstance(comp_colors, list):
                color = next(ax._get_lines.prop_cycler)['color']
            else:
                color = comp_colors.pop()
            colors.append(color)
            default_colors.append(color)
    for i in range(len(ssb_list)):
        color_index = 0
        for index in comp_group_index:
            if index != -1:
                ssb_colors.append(group_comp_colors[index])
            else:
                ssb_colors.append(default_colors[color_index])
                color_index += 1

    # plot data, fits, and components
    x= x[::-1]
    plt.plot(x, y[::-1], color=data_color, label='data')
    plt.plot(x, out.best_fit[::-1], '-', label='fit', color=fit_color, linewidth=1)
    if plot_init_fit:
        plt.plot(x, init[::-1], '--', label='init fit', color=init_fit_color)
    for i, comp_name in enumerate(comp_names):
        # print(comp_name+'_')
        plt.plot(x, comps[comp_name+'_'][::-1], '--', label=comp_labels[i], color=colors[i])
    if fit_ssb:
        for i, ssb_comp_name in enumerate(ssb_comp_names):
            plt.plot(x, comps[ssb_comp_name][::-1], '--', color=ssb_colors[i])
    if element and nucleus:
        nmr_nucleus = '$^{%d}$%s' % (nucleus, element)
    else:
        nmr_nucleus = ''
    plt.xlabel('\u03B4{} / ppm'.format(nmr_nucleus))
    plt.xlim(fit_range)
    if show_lgd:
        lgnd = plt.legend(loc=lgd_loc, labelspacing=0.2, fontsize=lgd_fsize, handlelength=1, frameon=False)
        # for line in lgnd.get_lines():
        #     line.set_linewidth(2)
        lgnd.set_draggable(True)
    if fig_save_dir:
        plt.savefig(fig_save_dir + save_name + '-fit' + '.png')
    if fit_save_dir:
        fit_vals_dict = out.best_values
        fitx = x
        fity = out.best_fit[::-1]
        save_obj = [fit_vals_dict, comps, fitx, fity, fit_ssb, ssb_list, fit_range, comp_names, comp_groups, group_names, ssb_comp_names]
        file = open(fit_save_dir + save_name, 'wb')
        pickle.dump(save_obj, file)
    if show_plot:
        plt.show()
    plt.close()
    return [x, y[::-1], out, groupless_amplitudes, group_amplitudes]

# To install package locally:
# python -m pip install -e .

# To deploy package
# python -m build
# twine upload dist/*

# remember to update version and to delete old files from dist/