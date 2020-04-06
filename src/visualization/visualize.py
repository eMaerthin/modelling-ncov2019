
from matplotlib import pyplot as plt
import pandas as pd

from src.models.schemas import *
from src.models.defaults import *
from src.models.states_and_functions import *


class Visualize:
    def __init__(self, params, df_progression_times, df_infections, df_individuals, expected_case_severity, logger):
        self._params = params
        self.df_progression_times = df_progression_times
        self.df_infections = df_infections
        self.df_individuals = df_individuals
        self.serial_interval_median = None
        self.fear = None
        self.active_people = None

        self._max_time_offset = 0
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            self._max_time_offset = np.inf

        self.detected_cases = None
        self._expected_case_severity = expected_case_severity
        self.logger = logger
        self._all_runs_detected = []
        self._all_runs_prevalence = []
        self._all_runs_severe = []

    @property
    def xlim(self):
        return (self._params.get(PLOT_XLIM_LEFT, default_plot_xlim_left),
                self._params.get(PLOT_XLIM_RIGHT, default_plot_xlim_right))

    @property
    def ylim(self):
        return (self._params.get(PLOT_YLIM_BOTTOM, default_plot_ylim_bottom),
                self._params.get(PLOT_YLIM_TOP, default_plot_ylim_top))

    @property
    def xlim_cut(self):
        left = self._params.get(PLOT_XLIM_CUT_LEFT, None)
        right = self._params.get(PLOT_XLIM_CUT_RIGHT, None)
        if left is None:
            left = self.xlim[0]
        if right is None:
            right = self.xlim[1]
        return (left, right)

    @property
    def ylim_cut(self):
        bottom = self._params.get(PLOT_YLIM_CUT_BOTTOM, None)
        top = self._params.get(PLOT_YLIM_CUT_TOP, None)
        if bottom is None:
            bottom = self.ylim[0]
        if top is None:
            top = self.ylim[1]
        return (bottom, top)

    def visualize_scenario(self, simulation_output_dir):
        fitting_successes = self.test_detected_cases(simulation_output_dir)
        q_ = self._params[DETECTION_MILD_PROBA]
        rstar_out = 2.34 * self._params[TRANSMISSION_PROBABILITIES][CONSTANT]
        c = self._params[FEAR_FACTORS].get(CONSTANT, self._params[FEAR_FACTORS][DEFAULT])[LIMIT_VALUE]
        fitting_successes_str = f'q,rstar,c,successes\n{q_},{rstar_out},{c},{fitting_successes}\n'
        fitting_successes_log_file = os.path.join(simulation_output_dir, 'fitting_successes.txt')
        with open(fitting_successes_log_file, "w") as out_fitting:
            out_fitting.write(fitting_successes_str)
        self.test_lognormal_prevalence(simulation_output_dir)
        self.test_lognormal_detected(simulation_output_dir)
        self.test_lognormal_severe(simulation_output_dir)

    def visualize_simulation(self, simulation_output_dir, serial_interval, fear, active_people, max_time_offset,
                             detected_cases):
        self.serial_interval_median = serial_interval
        self.fear = fear
        self.active_people = active_people
        self._max_time_offset = max_time_offset
        self.detected_cases = detected_cases

        serial_interval_median = self.serial_interval_median
        hack = self._params[EXPERIMENT_ID]
        c = self._params[TRANSMISSION_PROBABILITIES][CONSTANT]
        c_norm = c * self._params[AVERAGE_INFECTIVITY_TIME_CONSTANT_KERNEL]
        det = self._params[DETECTION_MILD_PROBA] * 100
        reduced_r = c_norm * self.fear(CONSTANT)

        self._params[EXPERIMENT_ID] = f'{self._params[EXPERIMENT_ID]}\n(median serial interval: {serial_interval_median:.2f} days, R*: {c_norm:.3f}'
        if self._params[TURN_ON_DETECTION]:
            self._params[EXPERIMENT_ID] = f'{self._params[EXPERIMENT_ID]}, Det: {det:.1f}%)'
        else:
            self._params[EXPERIMENT_ID] = f'{self._params[EXPERIMENT_ID]})'
        if self._params[FEAR_FACTORS].get(CONSTANT, self._params[FEAR_FACTORS][DEFAULT])[FEAR_FUNCTION] != FearFunctions.FearDisabled:
            self._params[EXPERIMENT_ID] = f'{self._params[EXPERIMENT_ID]}\n reduction factor: {(1 - self.fear(CONSTANT)):.3f}, reduced R*: {reduced_r:.3f}'

        self.lancet_store_graphs(simulation_output_dir)
        self.lancet_store_bins(simulation_output_dir)
        self.store_bins(simulation_output_dir)
        #self.store_bins_pl(simulation_output_dir)
        self.store_graphs(simulation_output_dir)
        self.store_detections(simulation_output_dir)
        self.store_semilogy(simulation_output_dir)
        self.doubling_time(simulation_output_dir)
        self.lancet_icu_beds(simulation_output_dir)
        self.icu_beds(simulation_output_dir)
        self.lancet_draw_death_age_cohorts(simulation_output_dir)
        self._params[EXPERIMENT_ID] = hack

    def doubling_time(self, simulation_output_dir):
        def doubling(x, y, window=100):
            x1 = x[:-window]
            x2 = x[window:]
            y1 = y[:-window]
            y2 = y[window:]
            a = (x2 - x1) * np.log(2)
            b = np.log(y2 / y1)
            c = a / b
            return c  # (x2 - x1) * np.log(2) / np.log(y2 / y1)

        def plot_doubling(x, ax, label, window=100):
            if len(x) > window:
                xval = x[:-window]
                yval = doubling(x.values, np.arange(1, 1 + len(x)))
                ax.plot(xval[yval < 28], yval[yval < 28], label=label)
                return True
            return False

        fig, ax = plt.subplots(nrows=1, ncols=1)
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        vals = df_r2.contraction_time.sort_values()
        plot_doubling(vals, ax, label='Trend line for prevalence')
        cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        plot_doubling(cond1, ax, label='Trend line for # imported cases')
        plot_doubling(cond2, ax, label='Trend line for Infected through constant kernel')
        plot_doubling(cond3, ax, label='Trend line for Infected through household kernel')
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        plot_doubling(ho_cases, ax, label='Trend line for # hospitalized cases')
        plot_doubling(d_cases, ax, label='Trend line for # deceased cases')
        ax.legend(loc='lower right')  # legend, loc='upper left')
        ax.set_title(f'Doubling times for simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'doubling_times.png'))
        plt.close(fig)


    def lancet_draw_death_age_cohorts(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        df_in = self.df_individuals
        lims = default_age_cohorts_with_descriptions

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for limm, limM, descr in lims:
            cond1 = df_in.age >= limm
            cond2 = df_in.age < limM
            cond = np.logical_and(cond1, cond2)
            filtered = df_r1.loc[df_r1.index.isin(df_in[cond].index)]
            death_cases = filtered[~filtered.tdeath.isna()].sort_values(by='tdeath').tdeath
            d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            d_times = np.arange(1, 1 + len(d_cases))
            ax.plot(np.append(d_cases, df_r2.contraction_time.max(axis=0)),
                    np.append(d_times, len(d_cases)), label=descr)

        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'lancet_supplementary_deceased_cases_age_analysis.png'))
        plt.close(fig)


    def draw_death_age_cohorts(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        df_in = self.df_individuals
        lims = default_age_cohorts_with_descriptions

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for limm, limM, descr in lims:
            cond1 = df_in.age >= limm
            cond2 = df_in.age < limM
            cond = np.logical_and(cond1, cond2)
            filtered = df_r1.loc[df_r1.index.isin(df_in[cond].index)]
            death_cases = filtered[~filtered.tdeath.isna()].sort_values(by='tdeath').tdeath
            d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            d_times = np.arange(1, 1 + len(d_cases))
            ax.plot(np.append(d_cases, df_r2.contraction_time.max(axis=0)),
                    np.append(d_times, len(d_cases)), label=descr)

        ax.legend()
        experiment_id = self._params[EXPERIMENT_ID]
        ax.set_title(f'cumulative deceased cases per age group \n {experiment_id}')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'deceased_cases_age_analysis.png'))
        plt.close(fig)


    def lancet_store_bins(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax0 = plt.subplots()
        r2_max_time = df_r2.contraction_time.max()
        if self.active_people < 10:
            if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                if self._max_time_offset != np.inf:
                    r2_max_time -= self._max_time_offset
            ax0.plot([r2_max_time], [0], 'ro', markersize=5, label='Last reported infection time')

        bins = np.arange(np.minimum(730, int(1 + r2_max_time)))
        cond3 = df_r2.contraction_time.sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                cond3 -= self._max_time_offset
        legend = []
        arr = []
        if len(cond3) > 0:
            arr.append(cond3)
            legend.append('Infections')
            ax0.hist(arr, bins, histtype='bar', stacked=False, label=legend)
        arr = []
        legend = []
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= r2_max_time].sort_values()
        if len(ho_cases) > 0:
            arr.append(ho_cases)
            legend.append('Hospitalized')
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                ho_cases -= self._max_time_offset

        ax0.hist(arr, bins, histtype='bar', stacked=False, label=legend)
        ax0.legend()
        ax0.set_ylabel('Incidents')
        ax0.set_xlabel('Time in days')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'lancet_bins.png'))
        plt.close(fig)


    def store_bins_pl(self, simulation_output_dir):
        # font = {'family': 'arial',
        #        'weight': 'regular',
        #        'size': 16}

        # matplotlib.rc('font', **font)
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        fig, (ax2, ax1, ax0) = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
        r2_max_time = df_r2.contraction_time.max()
        detected = df_r1.dropna(subset=['tdetection']).sort_values(by='tdetection').tdetection
        xloc = [3, 8, 13, 18, 23, 28]
        dates = ['13/03/20', '18/03/20', '23/03/20', '28/03/20', '2/03/20', '7/04/20']
        bins = np.arange(np.minimum(730, int(1 + r2_max_time)))
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                cond2 -= self._max_time_offset - 3
                cond3 -= self._max_time_offset - 3
        legend = []
        arr = []
        if len(cond2) > 0:
            arr.append(cond2)
            legend.append('Zarażenia poza domem')
        if len(cond3) > 0:
            arr.append(cond3)
            legend.append('Zarażenia w gosp. domowym')
        values, _, _ = ax0.hist(arr, bins, histtype='bar', stacked=True, label=legend, color=['blue', 'grey'])
        # ax0.plot([3]*2, [0, np.amax(values)], 'k-', label='Ogłoszenie zamknięcia granic Polski 13/03/2020')
        ax0.legend()
        ax0.set_ylabel('Zainfekowani')
        ax0.set_xlabel('Data')
        ax0.set_xticks(xloc)
        ax0.set_xticklabels(dates)
        ax0.set_xlim([0, 30])

        arr = []
        legend = []
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= r2_max_time].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= r2_max_time].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                ho_cases -= self._max_time_offset - 3
                d_cases -= self._max_time_offset - 3
        if len(d_cases) > 0:
            arr.append(d_cases)
            legend.append('Przypadki śmiertelne')
        if len(ho_cases) > 0:
            arr.append(ho_cases)
            legend.append('Hospitalizowani')
        values, _, _ = ax1.hist(arr, bins, histtype='bar', stacked=True, label=legend, color=['red', 'orange'])
        # ax1.plot([3]*2, [0, np.amax(values)], 'k-', label='Ogłoszenie zamknięcia granic Polski 13/03/2020')
        ax1.set_xlim([0, 30])

        ax1.set_ylabel('Przypadki poważne')
        ax1.set_xlabel('Data')
        ax1.legend()

        ax1.set_xticks(xloc)
        ax1.set_xticklabels(dates)

        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                det_cases -= self._max_time_offset - 3
        values, _, _ = ax2.hist(det_cases, bins, histtype='bar', stacked=True, label='Zdiagnozowani', color='green')
        # ax2.plot([3]*2, [0, np.amax(values)], 'k-', label='Ogłoszenie zamknięcia granic Polski 13/03/2020')
        ax2.set_xlim([0, 30])
        ax2.set_ylabel('Zdiagnozowani')
        ax2.set_xlabel('Data')

        ax2.set_xticks(xloc)
        ax2.set_xticklabels(dates)
        # ax2.legend()

        fig.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(simulation_output_dir, 'bins_report_pl.png'), dpi=300)
        plt.close(fig)


    def store_bins(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        if self._params[TURN_ON_DETECTION]:
            fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1)
        else:
            fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
        r2_max_time = df_r2.contraction_time.max()
        if self.active_people < 10:
            if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                if self._max_time_offset != np.inf:
                    r2_max_time -= self._max_time_offset
            ax0.plot([r2_max_time], [0], 'ro', markersize=5, label='Last reported infection time')

        bins = np.arange(np.minimum(730, int(1 + r2_max_time)))
        cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                cond1 -= self._max_time_offset
                cond2 -= self._max_time_offset
                cond3 -= self._max_time_offset
        legend = []
        arr = []
        if len(cond1) > 0:
            arr.append(cond1)
            legend.append('Imported')
        if len(cond2) > 0:
            arr.append(cond2)
            legend.append('Constant kernel')
        if len(cond3) > 0:
            arr.append(cond3)
            legend.append('Household')
        ax0.hist(arr, bins, histtype='bar', stacked=True, label=legend)
        ax0.legend()
        arr = []
        legend = []
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= r2_max_time].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= r2_max_time].sort_values()
        recovery_cases = df_r1[~df_r1.trecovery.isna()].sort_values(by='trecovery').trecovery
        r_cases = recovery_cases[recovery_cases <= r2_max_time].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                ho_cases -= self._max_time_offset
                d_cases -= self._max_time_offset
                r_cases -= self._max_time_offset
        if len(d_cases) > 0:
            arr.append(d_cases)
            legend.append('Deceased')
        if len(ho_cases) > 0:
            arr.append(ho_cases)
            legend.append('Hospitalized')
        if len(r_cases) > 0:
            arr.append(r_cases)
            legend.append('Recovered')
        ax1.hist(arr, bins, histtype='bar', stacked=True, label=legend)
        ax1.legend()
        ax0.set_title(f'Daily stacked summaries of simulated covid19\n {self._params[EXPERIMENT_ID]}')
        ax0.set_ylabel('Infections')
        ax0.set_xlabel('Time in days')
        ax1.set_ylabel('Outcome')
        ax1.set_xlabel('Time in days')
        if self._params[TURN_ON_DETECTION]:
            detected_cases = self.detected_cases(df_r1)
            det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                if self._max_time_offset != np.inf:
                    det_cases -= self._max_time_offset

            ax2.hist(det_cases, bins, histtype='bar', stacked=True, label='Daily officially detected cases')
            ax2.set_ylabel('Detections')
            ax2.set_xlabel('Time in days')
            ax2.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'bins.png'))
        plt.close(fig)


    def plot_values(self, values, label, ax, yvalues=None, type='plot', reduce_offset=True, dots=False):
        if len(values) > 0:
            x = values
            if reduce_offset:
                if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                    if self._max_time_offset != np.inf:
                        x -= self._max_time_offset
                    else:
                        return np.arange(0), np.arange(0)
            if yvalues is None:
                y = np.arange(1, 1 + len(x))
            else:
                y = yvalues
            if type == 'plot':
                if dots:
                    ax.plot(x, y, 'ok', label=label)
                else:
                    ax.plot(x, y, label=label)
            elif type == 'semilogy':
                ax.semilogy(x, y, label=label)
            if self._params[USE_TODAY_MARK]:
                today = float(self._params[TODAY_OFFSET])
                counter = sum(np.array(x) <= today)
                label_at_today = f'{label} at T={today}: {counter}'
                ax.plot([self._params[TODAY_OFFSET]] * 2, [0, len(values)], 'k-', label=label_at_today)
            return x, y
        return np.arange(0), np.arange(0)


    def store_graphs(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax = plt.subplots(nrows=1, ncols=1)
        vals = df_r2.contraction_time.sort_values()
        self.plot_values(vals, 'Prevalence', ax)
        #cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        #cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        #cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        #self.plot_values(cond1, 'Imported', ax)
        #self.plot_values(cond2, 'Inf. through constant kernel', ax)
        #self.plot_values(cond3, 'Inf. through household', ax)
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        self.plot_values(d_cases, 'Deceased', ax)
        self.plot_values(ho_cases, 'Hospitalized', ax)
        self.plot_values(det_cases, 'Detected', ax)
        self.add_observed_curve(ax)

        if QUARANTINE in df_r1.columns:
            quarantined_cases = df_r1[~df_r1.quarantine.isna()].sort_values(by='quarantine').quarantine
            q_cases = quarantined_cases[quarantined_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            self.plot_values(q_cases, 'Quarantined', ax)

        ax.legend()
        ax.set_title(f'simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        if self._params[FEAR_FACTORS].get(CONSTANT, self._params[FEAR_FACTORS][DEFAULT])[FEAR_FUNCTION] != FearFunctions.FearDisabled:
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

            ax2.set_ylabel('fear')  # we already handled the x-label with ax1
            detected = np.arange(0, 1 + len(det_cases)) #.cumsum().values
            yvals = []
            kernel_id = CONSTANT
            x = [0] + list(det_cases)

            """
            # TODO: this should be implemented in different way
            for t_, de_ in zip(x, detected):

                yvals.append(self.fear_fun[kernel_id](de_, 0, t_, self.fear_weights_detected[kernel_id],
                                                      self.fear_weights_deaths[kernel_id],
                                                      self.fear_loc[kernel_id],
                                                      self.fear_scale[kernel_id],
                                                      self.fear_limit_value[kernel_id]))
            #if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            #    if self._max_time_offset != np.inf:
            #        x = [elem - self._max_time_offset for elem in x]
            """
            ax2.plot(x, yvals, 'k--')
            ax2.tick_params(axis='y')
            ax2.set_ylim(bottom=0, top=1)
            ax2.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'summary.png'))
        plt.close(fig)


    def store_detections(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()

        self._all_runs_detected.append(det_cases)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title(f'detected cases in time\n {self._params[EXPERIMENT_ID]}')

        self.plot_values(det_cases, 'Detected', ax)
        self.add_observed_curve(ax)
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'summary_detections.png'))
        plt.close(fig)


    def lancet_store_graphs(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax = plt.subplots(nrows=1, ncols=1)
        vals = df_r2.contraction_time.sort_values()
        self.plot_values(vals, 'Prevalence', ax)
        self._all_runs_prevalence.append(vals)
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2

        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        self._all_runs_severe.append(ho_cases)
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        self.plot_values(d_cases, 'Deceased', ax)
        self.plot_values(ho_cases, 'Hospitalized', ax)
        self.plot_values(det_cases, 'Detected', ax)

        if QUARANTINE in df_r1.columns:
            quarantined_cases = df_r1[~df_r1.quarantine.isna()].sort_values(by='quarantine').quarantine
            q_cases = quarantined_cases[quarantined_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            self.plot_values(q_cases, 'Quarantined', ax)

        ax.legend()
        #ax.set_title(f'simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        if self._params[FEAR_FACTORS].get(CONSTANT, self._params[FEAR_FACTORS][DEFAULT])[FEAR_FUNCTION] != FearFunctions.FearDisabled:
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

            ax2.set_ylabel('fear')  # we already handled the x-label with ax1
            detected = np.arange(0, 1 + len(det_cases)) #.cumsum().values
            yvals = []
            kernel_id = CONSTANT
            x = [0] + list(det_cases)
            """
            for t_, de_ in zip(x, detected):
                yvals.append(self.fear_fun[kernel_id](de_, 0, t_, self.fear_weights_detected[kernel_id],
                                                      self.fear_weights_deaths[kernel_id],
                                                      self.fear_loc[kernel_id],
                                                      self.fear_scale[kernel_id],
                                                      self.fear_limit_value[kernel_id]))
            #if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            #    if self._max_time_offset != np.inf:
            #        x = [elem - self._max_time_offset for elem in x]
            """
            ax2.plot(x, yvals, 'k--')
            ax2.tick_params(axis='y')
            ax2.set_ylim(bottom=0, top=1)
            ax2.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'lancet_summary.png'))
        plt.close(fig)


    def store_semilogy(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        vals = df_r2.contraction_time.sort_values()
        if self.experimental_ub is None:
            self.experimental_ub = vals
        else:
            self.experimental_ub = np.minimum(vals, self.experimental_ub)
        if self.experimental_lb is None:
            self.experimental_lb = vals
        else:
            self.experimental_lb = np.maximum(vals, self.experimental_lb)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        self.plot_values(vals, 'Prevalence', ax, type='semilogy')
        #cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        #cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        #cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()

        #self.plot_values(cond1, 'Imported', ax, type='semilogy')
        #self.plot_values(cond2, 'Inf. through constant kernel', ax, type='semilogy')
        #self.plot_values(cond3, 'Inf. through household', ax, type='semilogy')

        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()

        self.plot_values(d_cases, 'Deceased', ax, type='semilogy')
        self.plot_values(ho_cases, 'Hospitalized', ax, type='semilogy')
        self.plot_values(det_cases, 'Detected', ax, type='semilogy')
        self.add_observed_curve(ax)

        if QUARANTINE in df_r1.columns:
            quarantined_cases = df_r1[~df_r1.quarantine.isna()].sort_values(by='quarantine').quarantine
            q_cases = quarantined_cases[quarantined_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            self.plot_values(q_cases, 'Quarantined', ax, type='semilogy')

        ax.legend()
        ax.set_title(f'simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'summary_semilogy.png'))
        plt.close(fig)


    def test_bandwidth_plot(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        self.plot_values(self.experimental_ub, 'Prevalence UB', ax)
        self.plot_values(self.experimental_lb, 'Prevalence LB', ax)
        ax.legend()
        ax.set_title(f'Test of bandwidth plot (showing min/max across multiple runs)')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'test_bandwidth_plot_summary_semilogy.png'))
        plt.close(fig)


    def test_lognormal_prevalence(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, run in enumerate(self._all_runs_prevalence):
            self.plot_values(run, f'Run {i}', ax, reduce_offset=False, type='semilogy')
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_title(f'Sample paths of prevalence')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'test_lognormal_prevalence.png'))
        plt.close(fig)


    def test_lognormal_detected(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, run in enumerate(self._all_runs_detected):
            self.plot_values(run, f'Run {i}', ax, reduce_offset=False, type='semilogy')
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_title(f'Sample paths of detected cases')
        #ax.set_title(f'Analiza')
        #xloc = [0, 5, 10, 15, 20]
        #dates = ['12/03/20', '17/03/20', '22/03/20', '27/03/20', '1/04/20', '6/04/20']
        #ax.set_ylabel('Zdiagnozowani (skala logarytmiczna)')
        #ax.set_xlabel('Data')
        #ax.set_xticks(xloc)
        #ax.set_xticklabels(dates, rotation=30)
        fig.tight_layout()

        plt.savefig(os.path.join(simulation_output_dir, 'test_lognormal_detected.png'))
        plt.close(fig)


    def test_lognormal_severe(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, run in enumerate(self._all_runs_severe):
            self.plot_values(run, f'Run {i}', ax, reduce_offset=False, type='semilogy')
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_title(f'Sample paths of severe cases')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'test_lognormal_severe.png'))
        plt.close(fig)


    def test_detected_cases(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        x = []
        y = []
        p_x = []
        p_y = []
        successes = 0
        for i, (run, run_p) in enumerate(zip(self._all_runs_detected, self._all_runs_prevalence)):
            x_, y_ = self.plot_values(run.values, f'Run {i}', ax, reduce_offset=False)
            #x_ = x_.values
            #TODO
            if len(x_) > self._params[NUMBER_OF_DETECTED_AT_ZERO_TIME]:
                t0 = x_[self._params[NUMBER_OF_DETECTED_AT_ZERO_TIME]]
                filt_x = x_[x_ <= t0 - 7]
                if len(filt_x) > 0:
                    arg_tminus7 = np.argmax(filt_x)
                    if np.abs(y_[arg_tminus7] - self._params[LAID_CURVE]["-7"]) < 0.1 * self._params[LAID_CURVE]["-7"]:
                        x.extend(list(x_))
                        y.extend(list(y_))
                        p_x_ = run_p.values
                        p_y_ = np.arange(1, 1 + len(p_x_))
                        p_x.extend(p_x_)
                        p_y.extend(p_y_)
                        successes += 1
        self.logger.info(f'There are {successes} successes')
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_xlim(self.xlim_cut)
        ax.set_ylim(self.ylim_cut)
        reduction = (1 - self._params[FEAR_FACTORS].get(CONSTANT, self._params[FEAR_FACTORS][DEFAULT])[LIMIT_VALUE]) * 100
        R = 2.34 * self._params[TRANSMISSION_PROBABILITIES][CONSTANT]
        reducted = (100 - reduction) * R / 100
        title = f'Prognoza diagnoz (q={self._params[DETECTION_MILD_PROBA]:.1f}, redukcja R* z {R:.2f} o {reduction:.0f}% do {reducted:.2f})'

        title2 = f'Prognoza liczby zakażonych\n(q={self._params[DETECTION_MILD_PROBA]:.1f}, redukcja R* z {R:.2f} o {reduction:.0f}% do {reducted:.2f})'

        ax.set_title(title)

        #ax.set_title(f'Sample paths of detected cases')
        xloc = [0, 5, 10, 15, 20, 25, 28]
        dates = ['02/04/20', '07/04/20', '12/04/20', '17/04/20', '22/04/20', '27/04/20', '30/04/20']
        ax.set_ylabel('Zdiagnozowani')
        ax.set_xlabel('Data')

        ax.set_xticks(xloc)
        ax.set_xticklabels(dates, rotation=30)
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'detected_cases_pl.png'))
        plt.close(fig)
        if successes > 0:
            #xy = np.vstack([x, y])
            #z = scipy.stats.gaussian_kde(xy)(xy)
            fig, ax = plt.subplots()
            ax.set_title(title)

            #ax.scatter(x, y, c=z, s=1, edgecolor='')
            ax.scatter(x, y, s=1, edgecolor='')
            self.add_observed_curve(ax)
            xloc = [0, -5, -10, -15, -20]
            dates = ['02/04/20', '28/03/20', '23/03/20', '18/03/20', '13/03/20']
            ax.set_ylabel('Zdiagnozowani')
            ax.set_xlabel('Data')
            ax.set_xticks(xloc)
            ax.set_xticklabels(dates, rotation=30)
            ax.set_xlim([-20, 0])
            ax.set_ylim([0, self._params[NUMBER_OF_DETECTED_AT_ZERO_TIME]*1.4])

            fig.tight_layout()
            plt.savefig(os.path.join(simulation_output_dir, 'detected_cases_density_pl.png'))
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.set_title(title)

            #ax.scatter(x, y, c=z, s=1, edgecolor='')
            ax.scatter(x, y, s=1, edgecolor='')
            self.add_observed_curve(ax)

            ax.set_xlim(self.xlim_cut)
            ax.set_ylim(self.ylim_cut)
            xloc = [0, 5, 10, 15, 20, 25, 28]
            dates = ['02/04/20', '07/04/20', '12/04/20', '17/04/20', '22/04/20', '27/04/20', '30/04/20']
            ax.set_ylabel('Zdiagnozowani')
            ax.set_xlabel('Data')
            ax.set_xticks(xloc)
            ax.set_xticklabels(dates, rotation=30)
            fig.tight_layout()
            plt.savefig(os.path.join(simulation_output_dir, 'detected_cases_density_pl_cut.png'))
            plt.close(fig)
            #############################################
            #xy = np.vstack([p_x, p_y])
            #z = scipy.stats.gaussian_kde(xy)(xy)
            fig, ax = plt.subplots()
            ax.set_title(title2)

            #ax.scatter(p_x, p_y, c=z, s=1, edgecolor='')
            ax.scatter(p_x, p_y, s=1, edgecolor='')
            self.add_observed_curve(ax)
            xloc = [0, -5, -10, -15, -20]
            dates = ['02/04/20', '28/03/20', '23/03/20', '18/03/20', '13/03/20']
            ax.set_ylabel('Zakażeni')
            ax.set_xlabel('Data')
            ax.set_xticks(xloc)
            ax.set_xticklabels(dates, rotation=30)
            ax.set_xlim([-20, 0])
            ax.set_ylim([0, self._params[NUMBER_OF_DETECTED_AT_ZERO_TIME] * 6.0])

            fig.tight_layout()
            plt.savefig(os.path.join(simulation_output_dir, 'prevalence_density_pl.png'))
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.set_title(title2)

            #ax.scatter(p_x, p_y, c=z, s=1, edgecolor='')
            ax.scatter(p_x, p_y, s=1, edgecolor='')
            self.add_observed_curve(ax)

            ax.set_xlim(self.xlim_cut)
            ax.set_ylim(self.ylim_cut)
            xloc = [0, 5, 10, 15, 20, 25, 28]
            dates = ['02/04/20', '07/04/20', '12/04/20', '17/04/20', '22/04/20', '27/04/20', '30/04/20']
            ax.set_ylabel('Zakażeni')
            ax.set_xlabel('Data')
            ax.set_xticks(xloc)
            ax.set_xticklabels(dates, rotation=30)
            fig.tight_layout()
            plt.savefig(os.path.join(simulation_output_dir, 'prevalence_density_pl_cut.png'))
            plt.close(fig)
        return successes


    def add_observed_curve(self, ax):
        if self._params[LAID_CURVE].items():
            laid_curve_x = np.array([float(elem) for elem in self._params[LAID_CURVE].keys()])
            if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                if self._max_time_offset != np.inf:
                    print(self._max_time_offset)
                    print(self._params[LAID_CURVE].keys())
                    laid_curve_x = np.array([float(elem) + self._max_time_offset for elem in self._params[LAID_CURVE].keys()])
            laid_curve_y = np.array(list(self._params[LAID_CURVE].values()))
            self.plot_values(laid_curve_x, 'Cases observed in PL', ax, yvalues=laid_curve_y, dots=True)


    def lancet_icu_beds(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax = plt.subplots(nrows=1, ncols=1)
        cond = [k for k, v in self._expected_case_severity.items() if v == ExpectedCaseSeverity.Critical]
        critical = df_r1.loc[df_r1.index.isin(cond)]
        plus = critical.t2.values
        deceased = critical[~critical.tdeath.isna()]
        survived = critical[critical.tdeath.isna()]
        minus1 = survived.trecovery.values
        minus2 = deceased.tdeath.values
        max_time = df_r2.contraction_time.max(axis=0)
        df_plus = pd.DataFrame({'t': plus, 'd': np.ones_like(plus)})
        df_minus1 = pd.DataFrame({'t': minus1, 'd': -np.ones_like(minus1)})
        df_minus2 = pd.DataFrame({'t': minus2, 'd': -np.ones_like(minus2)})
        df = df_plus.append(df_minus1).append(df_minus2).sort_values(by='t')
        df = df[df.t <= max_time]
        if len(df) == 0:
            return
        cumv = df.d.cumsum().values
        x = df.t.values

        self.plot_values(x, yvalues=cumv, label='ICU required', ax=ax)

        largest_y = cumv.max()
        icu_availability = self._params[ICU_AVAILABILITY]

        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= max_time].sort_values()
        if len(d_cases) > 0:
            self.plot_values(d_cases, 'deceased', ax)
            largest_y = max(largest_y, len(d_cases))
        t = [0, max_time]
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                t = [elem - self._max_time_offset for elem in t]
        ax.plot(t, [icu_availability] * 2, label=f'ICU capacity ({icu_availability})')
        cumv_filter_flag = cumv > icu_availability
        if cumv[cumv_filter_flag].any():
            critical_t = df.t.values[cumv_filter_flag].min()
            self.band_time = critical_t

            ax.plot([critical_t] * 2, [0, largest_y], label=f'Critical time {critical_t:.1f}')
        ax.legend()  # 'upper left')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'lancet_icu_beds_analysis.png'))
        plt.close(fig)


    def icu_beds(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax = plt.subplots(nrows=1, ncols=1)
        cond = [k for k, v in self._expected_case_severity.items() if v == ExpectedCaseSeverity.Critical]
        critical = df_r1.loc[df_r1.index.isin(cond)]
        plus = critical.t2.values
        deceased = critical[~critical.tdeath.isna()]
        survived = critical[critical.tdeath.isna()]
        minus1 = survived.trecovery.values
        minus2 = deceased.tdeath.values
        max_time = df_r2.contraction_time.max(axis=0)
        df_plus = pd.DataFrame({'t': plus, 'd': np.ones_like(plus)})
        df_minus1 = pd.DataFrame({'t': minus1, 'd': -np.ones_like(minus1)})
        df_minus2 = pd.DataFrame({'t': minus2, 'd': -np.ones_like(minus2)})
        df = df_plus.append(df_minus1).append(df_minus2).sort_values(by='t')
        df = df[df.t <= max_time]
        if len(df) == 0:
            return
        cumv = df.d.cumsum().values
        x = df.t.values
        self.plot_values(x, yvalues=cumv, label='ICU required', ax=ax)

        largest_y = cumv.max()
        icu_availability = self._params[ICU_AVAILABILITY]

        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= max_time].sort_values()
        t = [0, max_time]
        ax.plot(t, [icu_availability] * 2, label='ICU capacity')
        cumv_filter_flag = cumv > icu_availability
        if cumv[cumv_filter_flag].any():
            critical_t = df.t.values[cumv_filter_flag].min()
            self.band_time = critical_t
            ax.plot([critical_t] * 2, [0, largest_y], label=f'Critical time {critical_t:.1f}')
        ax.legend() #'upper left')
        ax.set_title('ICU requirements\n{self._params[EXPERIMENT_ID]}')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'icu_beds_analysis.png'))
        plt.close(fig)