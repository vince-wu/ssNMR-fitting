import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmfit import Parameters
from lmfit.models import ExponentialModel, GaussianModel, PseudoVoigtModel
import pprint
from formatting import format_plot


data_dir = 'c:/Users/Vincent/Box/Clement Research/NMR/raw xy data/'
save_dir = 'c:/Users/Vincent/Box/Clement Research/NMR/fit summaries/'
def simple_parse_data(data_path):
    """
    DESCRIPTION: Parses a .txt file containing NMR spectra with frequency and intensity data and returns two arrays for each data column
    PARAMETERS: 
        data_path: path-like string
            the path to the raw .txt file
    RETURNS: [intensity, freq_ppm, freq_Hz, freq_ppm, nuclei, experiment, exp_name]
        intensity: numpy array
            a numpy array containing intensity data
        freq_Hz: numpy array
            a numpy array containing frequency data in Hz
        freq_ppm: numpy array
            a numpy array containing frequency data in freq_ppm
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

def generate_summary(model_result, comp_names, comp_groups, group_names, fit_ssb, ssb_list, ssb_comp_names):
    fit_vals_dict = model_result.best_values
    component_df = pd.DataFrame(columns=['component(s)', 'group', 'relative %', 'amplitude','center','sigma','fraction', 'fwhm','height'])
    comp_names = comp_names.copy()
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
            comp_names.append(ssb_comp_name.rstrip('_'))
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
    original_comp_group_names = comp_group_names.copy()
    if len(ssb_comp_names) > 0:
        for ssb_comp_name in ssb_comp_names:
            for i, original_comp_group_names in enumerate(original_comp_groups):
                if any(name in ssb_comp_name for name in original_comp_group_names):
                    comp_groups[i].append(ssb_comp_name)
    # print('comp_groups: ', comp_groups)
    # print('group names: ', group_names)
    # print('comp_group_names: ', comp_group_names)
    # print('comp names: ', comp_names)
    # print(len(comp_groups))
    for group, name in zip(comp_groups, group_names):
        total_amplitude = 0
        iso_amplitudes = []
        centers = []
        comps = ''
        for comp in group:
            comps += '{} '.format(comp)
            # getting center data, just for isotropic peaks
            for i in range(len(comp_names)):
                if comp == comp_names[i]:
                    iso_amplitude = comp_amplitudes[i]
                    center = comp_centers[i]
                    iso_amplitudes.append(iso_amplitude)
                    centers.append(center)
            # getting amplitude data for all peaks
            for i in range(len(comp_names)):
                if comp.rstrip('_') == comp_names[i]:
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
        comp_names.append(comps)
    group_relative_amps = [100*x/sum(comp_amplitudes) for x in comp_group_amplitudes]
    [comp_amplitudes.append(x) for x in comp_group_amplitudes]
    [comp_relative_amps.append(x) for x in group_relative_amps]
    comp_data = [comp_names, comp_group_names, comp_relative_amps, comp_amplitudes, comp_centers, comp_sigmas, comp_fractions, 
    comp_fwhms, comp_heights]
    for name, data in zip(list(component_df.columns), comp_data):
        # print(data)
        component_df[name] = data
        # print('added {}'.format(name))
    component_df.to_csv(save_dir + 'fit.csv')

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
        print('comp{}=\\'.format(i))
        pprint.pprint(comp)

def fit(data_file, fit_range, components_list, comp_constraints, comp_names, comp_groups=None, group_names=None,
        data_color='black', fit_color='red', init_fit_color='green', comp_colors=['blue', 'red'], plot_init_fit=True,
        fit_ssb=True, ssb_list=[], mas_freq=60000,
        show_lgd=True, lgd_loc=0, lgd_fsize=22):
    # get NMR data
    intensity, freq_Hz, freq_ppm, larmor_freq = simple_parse_data(data_file)
    # checking inputs
    if len(components_list) != len(comp_constraints):
        raise ValueError("Number of component constraints ({}) is not equal to the number of components ({})"\
            .format(len(comp_constraints), len(components_list)))
    # cut data to fit range
    lower_bound = next(x for x, val in enumerate(freq_ppm) if val <= fit_range[0])
    upper_bound = next(x for x, val in enumerate(freq_ppm) if val < fit_range[1])
    x = freq_ppm[lower_bound:upper_bound]
    y = intensity[lower_bound:upper_bound]
    # format plots
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
                        param.set(value=component[param_name])
                compiled_components.append(pseudo_voigt)
    # build model containing all generated components
    model = compiled_components[0]
    for i in range(1, len(compiled_components)):
        model = model + compiled_components[i]
    init = model.eval(pars, x=x)
    # fit components to data
    out = model.fit(y, pars, x=x)
    # print fitting results
    print(out.fit_report(min_correl=0.5))
    comps = out.eval_components(x=x)
    # print(out.best_values)
    x= x[::-1]
    plt.plot(x, y[::-1], color=data_color, label='data')
    plt.plot(x, out.best_fit[::-1], '-', label='fit', color=fit_color, linewidth=1)
    if plot_init_fit:
        plt.plot(x, init[::-1], '--', label='init fit', color=init_fit_color)
    # print fitting results as dictionary objects
    print_fit_vals(out, comp_names)
    # generate summary of fits
    generate_summary(out, comp_names, comp_groups, group_names, fit_ssb, ssb_list, ssb_comp_names)

    
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
            colors.append(comp_colors[index])
        else:
            color = next(ax._get_lines.prop_cycler)['color']
            colors.append(color)
            default_colors.append(color)
    for i in range(len(ssb_list)):
        color_index = 0
        for index in comp_group_index:
            if index != -1:
                ssb_colors.append(comp_colors[index])
            else:
                ssb_colors.append(default_colors[color_index])
                color_index += 1
    # print(comp_group_index)
    # print(colors)
    # plot data, fits, and components

    for i, comp_name in enumerate(comp_names):
        # print(comp_name+'_')
        plt.plot(x, comps[comp_name+'_'][::-1], '--', label=comp_labels[i], color=colors[i])
    if fit_ssb:
        for i, ssb_comp_name in enumerate(ssb_comp_names):
            plt.plot(x, comps[ssb_comp_name][::-1], '--', color=ssb_colors[i])
    
    plt.xlabel('\u03B4/ ppm')
    plt.xlim(fit_range)
    if show_lgd:
        lgnd = plt.legend(loc=lgd_loc, labelspacing=0.2, fontsize=lgd_fsize, handlelength=1, frameon=False)
        # for line in lgnd.get_lines():
        #     line.set_linewidth(2)
        lgnd.set_draggable(True)
    plt.show()

comp0=\
{'amplitude': 175945691513.5195,
 'center': 500.173691,
 'fraction': 0.25937517,
 'sigma': 239.5564594157353}
comp1=\
{'amplitude': 267854355812.6343,
 'center': 275.140072,
 'fraction': 0.00012542,
 'sigma': 156.9640161234177}
comp2=\
{'amplitude': 220460472222.7904,
 'center': 131.872497,
 'fraction': 0.04651641,
 'sigma': 110.4590220133286}
comp3=\
{'amplitude': 143296226890.10104,
 'center': 28.0754571,
 'fraction': 0.07565613,
 'sigma': 72.99218373146846}
comp4=\
{'amplitude': 51231276645.67333,
 'center': 6.45267436,
 'fraction': 0.99995192,
 'sigma': 12.130313897866142}

comp0_constraints = {
    'amplitude_vary' : True,
    'amplitude_min' : None,
    'amplitude_max' : None,
    'amplitude_expr' : None,
    'center_vary' : False,
    'center_min' : None,
    'center_max' : None,
    'center_expr' : None,
    'fraction_vary' : False,
    'fraction_min' : None,
    'fraction_max' : None,
    'fraction_expr' : None,
    'sigma_vary' : True,
    'sigma_min' : None,
    'sigma_max' : None,
    'sigma_expr' : None,
}
comp1_constraints = {
    'amplitude_vary' : True,
    'amplitude_min' : None,
    'amplitude_max' : None,
    'amplitude_expr' : None,
    'center_vary' : True,
    'fraction_vary' : True,
}
comp2_constraints = {
    'amplitude_vary' : True,
    'amplitude_min' : None,
    'amplitude_max' : None,
    'amplitude_expr' : None,
    'center_vary' : True,
    'fraction_vary' : True,
}
comp3_constraints = {
    'amplitude_vary' : True,
    'amplitude_min' : None,
    'amplitude_max' : None,
    'amplitude_expr' : None,
    'center_vary' : True,
    'fraction_vary' : True,
}
comp4_constraints = {
    'amplitude_vary' : True,
    'amplitude_min' : None,
    'amplitude_max' : None,
    'amplitude_expr' : None,
    'center_vary' : True,
    'fraction_vary' : True,
}

comp_names = ['p1', 'p2', 'p3', 'p4', 'd1']
para_comps = ['p1', 'p2', 'p3', 'p4']
dia_comps = ['d1']
comp_groups = [para_comps, dia_comps]
group_names = ['paramagnetic', 'diamagnetic']
components = [comp0, comp1, comp2, comp3, comp4]
constraints = [comp0_constraints, comp0_constraints, comp0_constraints, comp0_constraints, comp0_constraints]
fit(
    data_file=data_dir+'MnTi-B10-S13-7Li-d1-20ms.txt',
    fit_range=(4000, -4000),
    components_list=components,
    comp_constraints=constraints,
    comp_names=comp_names,
    comp_groups = comp_groups,
    group_names = group_names,
    comp_colors=['blue', 'red'],
    fit_ssb=True,
    ssb_list=[1, -1],
    mas_freq=60000,
    plot_init_fit=True
    )