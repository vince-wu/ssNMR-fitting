import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmfit import Parameters
from lmfit.models import ExponentialModel, GaussianModel, PseudoVoigtModel
import pprint
from formatting import format_plot


data_dir = 'c:/Users/Vincent/Box/Clement Research/NMR/raw xy data/'
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
    # print(freq_ppm)
    # print(intensity)
    return [intensity, freq_Hz, freq_ppm]

def fit(data_file, fit_range, components_list, comp_constraints, comp_color='blue'):
    intensity, freq_Hz, freq_ppm = simple_parse_data(data_file)
    # cut data to fit range
    lower_bound = next(x for x, val in enumerate(freq_ppm) if val <= fit_range[0])
    upper_bound = next(x for x, val in enumerate(freq_ppm) if val < fit_range[1])
    plt, ax = format_plot(
        fig_size=(8,8),
    )
    x = freq_ppm[lower_bound:upper_bound][::-1]
    y = intensity[lower_bound:upper_bound][::-1]
    # plt.plot(x,y)
    # plt.show()
    # print(x)
    # return
    compiled_components = []
    pars = Parameters()
    comp_names = []
    for i in range(len(components_list)):
        component = components_list[i]
        constraints = comp_constraints[i]
        # initial guesses
        gauss_to_lorenzt_ratio = component['fraction']
        center = component['center']
        sigma = component['sigma']
        amp = component['amplitude']
        # constraints
        amp_vary = constraints['amplitude_vary']
        amp_min = constraints['amplitude_min']
        amp_max = constraints['amplitude_max']
        amp_expr = constraints['amplitude_expr']
        prefix = 'comp{}_'.format(i)
        pseudo_voigt = PseudoVoigtModel(prefix=prefix)
        pars.update(pseudo_voigt.make_params())
        params_list = ['fraction', 'center', 'sigma', 'amplitude']
        constraints_list = ['vary', 'min', 'max', 'expr']
        for param_name in params_list:
            param = pars['{}{}'.format(prefix, param_name)]
            param.set(value=component[param_name])
            for constraint_name in constraints_list:
                param_constrain_name = '{}_{}'.format(param_name, constraint_name)
                if param_constrain_name in constraints.keys():
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
        comp_names.append(prefix)
    
    model = compiled_components[0]
    for i in range(1, len(compiled_components)):
        model = model + compiled_components[i]
    init = model.eval(pars, x=x)
    out = model.fit(y, pars, x=x)
    print(out.fit_report(min_correl=0.5))

    plt.plot(x, y)
    plt.plot(x, init, '--', label='init fit')
    plt.plot(x, out.best_fit, '-', label='best fit')
    comps = out.eval_components(x=x)
    for i, comp_name in enumerate(comp_names):
        comp = {
            'amplitude': pars['{}{}'.format(comp_name, 'amplitude')].value,
            'center': pars['{}{}'.format(comp_name, 'center')].value,
            'sigma': pars['{}{}'.format(comp_name, 'sigma')].value,
            'fraction': pars['{}{}'.format(comp_name, 'fraction')].value,
        }
        print('comp{}=\\'.format(i))
        pprint.pprint(comp)
    for comp_name in comp_names:
        plt.plot(x, comps[comp_name], '--', label='component {}'.format(comp_name))
    plt.legend()
    plt.show()


comp0=\
{'amplitude': 169590000000.0,
 'center': 500.173691,
 'fraction': 0.25937517,
 'sigma': 238.328096}
comp1=\
{'amplitude': 272680000000.0,
 'center': 275.140072,
 'fraction': 0.00012542,
 'sigma': 162.910728}
comp2=\
{'amplitude': 233960000000.0,
 'center': 131.872497,
 'fraction': 0.04651641,
 'sigma': 115.098142}
comp3=\
{'amplitude': 120230000000.0,
 'center': 28.0754571,
 'fraction': 0.07565613,
 'sigma': 73.3100404}
comp4=\
{'amplitude': 53730000000.0,
 'center': 6.45267436,
 'fraction': 0.99995192,
 'sigma': 11.9608628}
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
}
comp2_constraints = {
    'amplitude_vary' : True,
    'amplitude_min' : None,
    'amplitude_max' : None,
    'amplitude_expr' : None,
}
comp3_constraints = {
    'amplitude_vary' : True,
    'amplitude_min' : None,
    'amplitude_max' : None,
    'amplitude_expr' : None,
}
comp4_constraints = {
    'amplitude_vary' : True,
    'amplitude_min' : None,
    'amplitude_max' : None,
    'amplitude_expr' : None,
}
components = [comp0, comp1, comp2, comp3, comp4]
constraints = [comp0_constraints, comp0_constraints, comp0_constraints, comp0_constraints, comp0_constraints]
fit(
    data_file=data_dir+'MnTi-B10-S13-7Li-d1-20ms.txt',
    fit_range=(1200, -300),
    components_list=components,
    comp_constraints=constraints
    )