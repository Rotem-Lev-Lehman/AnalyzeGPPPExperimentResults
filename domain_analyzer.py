import pandas as pd
import numpy as np
import decimal
import matplotlib.pyplot as plt


def drange(x, y, jump):
    """ A generator for sorted values in the range: [x, y], with jumps of jump.

    :param x: the lowest value generated
    :type x: float
    :param y: the highest value generated
    :type y: float
    :param jump: the jumps between values generated
    :type jump: str
    :return: the generator
    """
    while x <= y:
        yield float(x)
        x += decimal.Decimal(jump)


def get_regular_plot_styles_dict():
    """ Returns the graph plotting style for regular domains (with differences between the methods).

    :return: the style dictionary in the format of:
            {solver_type: {'style':line_style, 'color':line_color, 'width':line_width}}
    :rtype: dict
    """
    styles_dict = {'m1': {'style': '-', 'color': 'black', 'width': 3},
                   'm2': {'style': ':', 'color': 'red', 'width': 3},
                   'm3': {'style': '-', 'color': 'green', 'width': 2},
                   'm4': {'style': '--', 'color': 'purple', 'width': 2},
                   'Random': {'style': '-', 'color': 'blue', 'width': 2},
                   'Random_upper_bound': {'style': '--', 'color': 'blue', 'width': 1},
                   'Random_lower_bound': {'style': '--', 'color': 'blue', 'width': 1},
                   'Hindsight': {'style': '--', 'color': 'yellow', 'width': 2}
                   }
    return styles_dict


def get_solo_solver_plot_styles_dict():
    """ Returns the graph plotting style for domains without differences between the methods.

    :return: the style dictionary in the format of:
            {solver_type: {'style':line_style, 'color':line_color, 'width':line_width}}
    :rtype: dict
    """
    styles_dict = {'m1': {'style': '-', 'color': 'blue', 'width': 2},
                   'Hindsight': {'style': '--', 'color': 'yellow', 'width': 2}
                   }
    return styles_dict


class DomainAnalyzer:

    def __init__(self, planner_type, domain_name, same_graph_line=False,
                 add_legend=False, truncate_x_axis_at=None, save_figure=True):
        """ Initializes a new DomainAnalyzer instance

        :param planner_type: the type of the planner we are analyzing now. Can be 'MAFS', or 'Joint_Projection'
        :type planner_type: str
        :param domain_name: the name of the domain we are analyzing now
        :type domain_name: str
        :param same_graph_line: indicates whether we want to show all methods as the same line (True),
                                or each with a different line (False). Defaults to False
        :type same_graph_line: bool
        :param add_legend: indicates whether we want to show the legend in the graph (True),
                                or hide the legend (False). Defaults to False
        :type add_legend: bool
        :param truncate_x_axis_at: the max dependencies we want to show in the graph.
                                    If None - show all x-axis. Defaults to None
        :type truncate_x_axis_at: int
        :param save_figure: indicates whether we want to save the coverage figure (True),
                            or just to show it in here (False). Defaults to True
        :type save_figure: bool
        """
        self.planner_type = planner_type
        self.domain_name = domain_name
        self.same_graph_line = same_graph_line
        self.add_legend = add_legend
        self.truncate_x_axis_at = truncate_x_axis_at
        self.save_figure = save_figure

        self.solver2csv_file = {}  # {solver_type: csv_file_path}
        self.solver2data = {}  # {solver_type: data_for_solver(DataFrame)}
        self.problems = None  # set of all of the problem names of this domain
        self.solver2dict_problem2data = {}  # {solver_type: {problem_name: data_of_problem(DataFrame)}}
        self.solver2dict_problem2success_data = {}  # {solver_type: {problem_name: data_of_solved_problem(DataFrame)}}
        self.percentages = None  # all of the percentages of revealing dependencies.
        self.initialize_percentages()
        self.published_dep_axis_set = set()  # set of the x-axis of the graph (amount of published dependencies)
        self.problem2hindsight_val = {}  # {problem_name: lowest_amount_of_used_dep (hindsight)}
        # coverage graphs:
        self.solver2coverage_graph_data = {}  # {solver_type: coverage_graph_data(DataFrame)}
        self.hindsight_graph_data = None  # DataFrame of the hindsight graph
        self.joint_graph_coverage_data = None  # DataFrame of the joint graph's coverage score
        # random solver stuff:
        self.random_solver_name = None
        self.has_random_solver = False
        # coverage table:
        self.coverage_table = {}  # {solver_type: {Md%: min dependencies to get max coverage percentage from max dep,
        #                                            C: max coverage,
        #                                       C_last: the coverage when revealing all dependencies}}
        self.max_dep = None  # maximal amount of dependencies in all of the problems
        # cost table:
        self.cost_table_entry = {}  # {solver_type: {Min:..., Max:..., Min. dep:..., Max. dep:..., Imp.:...}}
        # optimal stuff:
        self.solver2dict_problem2first_solved = {}  # {solver_type: {problem_name: #min dependencies needed to solve}}

    def add_solver(self, solver_type, csv_file_path, random_solver):
        """ Adds a solver to the DomainAnalyzer.

        :param solver_type: the type of the solver
        :type solver_type: str
        :param csv_file_path: the path to the results of the solver
        :type csv_file_path: str
        :param random_solver: indicates whether the given solver is the random solver (True), or not (False).
        :type random_solver: bool
        """
        self.solver2csv_file[solver_type] = csv_file_path
        if random_solver:
            self.has_random_solver = True
            self.random_solver_name = solver_type

    def analyze_results(self, output_folder, draw_graph_for_domain):
        """ Analyzes the results given, and creates graphs and summarized DataFrames.

        :param output_folder: the folder path we want to save the graph to
        :type output_folder: str
        :param draw_graph_for_domain: indicates wheter we want to draw a coverage graph for this domain (True) or not (False)
        :type draw_graph_for_domain: bool
        """
        self.read_results()
        self.analyze_coverage(output_folder, draw_graph_for_domain)
        self.analyze_cost()

    def read_results(self):
        """ Reads all results files.
        """
        for solver_type, csv_file_path in self.solver2csv_file.items():
            self.read_results_file(solver_type, csv_file_path)

    def read_results_file(self, solver_type, csv_file_path):
        """ Reads a specific results file.

        :param solver_type: the solver type that made this results file
        :type solver_type: str
        :param csv_file_path: the path to the results file
        :type csv_file_path: str
        """
        df = pd.read_csv(csv_file_path)
        self.solver2data[solver_type] = df
        self.save_problems(solver_type, df)

    def save_problems(self, solver_type, df):
        """ Saves all the problem names and data of the domain.

        :param solver_type: the type of this solver
        :type solver_type: str
        :param df: the results DataFrame
        :type df: pd.DataFrame
        """
        '''
        Note the columns of the results file:
        column_names = ['Percentage of actions selected', ' folder name', ' success/failure',
       ' plan cost', ' plan make span', ' ? makespan plan time ?',
       ' total time', ' ? senderstate counter ?', ' ? state expend counter ?',
       ' ? generate counter ?', ' amount of dependencies used',
       ' amount of dependecies published']
        '''
        all_names = set(df[' folder name'].unique())
        if self.problems is None:
            self.problems = all_names
        elif self.problems != all_names:
            raise Exception('The problems must be the same in all of the solvers!')

        problem2data = {}
        problem2success_data = {}
        for problem in self.problems:
            # Save the entire sub-df:
            sub_df = df[df[' folder name'] == problem]
            sub_df.sort_values(by=['Percentage of actions selected'], inplace=True)
            problem2data[problem] = sub_df
            all_percentages = sorted(list(sub_df['Percentage of actions selected'].unique()))
            if all_percentages != self.percentages:
                raise Exception('The percentages must agree!')

            # Save only the sub-df of the rows that had success in solving the problem:
            success_sub_df = sub_df[sub_df[' success/failure'] == '  success']
            problem2success_data[problem] = success_sub_df

            # Save the x-axis of the graph:
            published_dep = set(sub_df[' amount of dependecies published'])
            used_dep = success_sub_df[' amount of dependencies used']  # this is for the calculation of the hindsight
            hindsight = used_dep.min()
            published_dep.add(hindsight)  # this will add the hindsight's graph to the x-axis as well
            self.published_dep_axis_set |= published_dep  # union the two sets together to create the axis

            # Save the hindsight:
            if problem not in self.problem2hindsight_val.keys():
                self.problem2hindsight_val[problem] = hindsight
            elif np.isnan(self.problem2hindsight_val[problem]) or self.problem2hindsight_val[problem] > hindsight:
                self.problem2hindsight_val[problem] = hindsight

        self.solver2dict_problem2data[solver_type] = problem2data
        self.solver2dict_problem2success_data[solver_type] = problem2success_data

    def analyze_coverage(self, output_folder, draw_graph_for_domain):
        """ Analyze the coverage of the results files for each of the solvers.

        :param output_folder: the folder path we want to save the graph in
        :type output_folder: str
        :param draw_graph_for_domain: indicates wheter we want to draw a coverage graph for this domain (True) or not (False)
        :type draw_graph_for_domain: bool
        """
        for solver_type in self.solver2data.keys():
            random_solver = self.has_random_solver and self.random_solver_name == solver_type
            self.analyze_coverage_for_solver(solver_type, random_solver)
            self.find_first_solved_for_each_problem(solver_type)
        self.calculate_hindsight_graph()
        self.analyze_summarized_coverage_results()
        if draw_graph_for_domain:
            self.create_coverage_graph(output_folder)
        self.create_coverage_table()

    def create_coverage_table(self):
        """ Creates a coverage table's entry for this domain and planner.
            The format of the table is specified in the paper.
        """
        self.max_dep = max(self.published_dep_axis_set)
        for column in self.joint_graph_coverage_data.columns:
            if self.has_random_solver and column == self.random_solver_name:
                # Do not add the random solver to the table
                continue
            Md = self.joint_graph_coverage_data[column].idxmax()
            C = self.joint_graph_coverage_data[column].max()
            C_last = self.joint_graph_coverage_data[column].iloc[-1]
            Md_percentage = Md / self.max_dep
            self.coverage_table[column] = {'Md%': Md_percentage, 'C': C, 'C_last': C_last}

    def create_coverage_graph(self, output_folder):
        """ Creates a coverage graph for this domain and planner.

        :param output_folder: the folder path we want to save the graph in
        :type output_folder: str
        """
        width1 = 4
        height1 = 2
        width_height_1 = (width1, height1)

        plt.figure(figsize=width_height_1)

        if not self.same_graph_line:
            styles_dict = get_regular_plot_styles_dict()
        else:
            styles_dict = get_solo_solver_plot_styles_dict()

        for col in styles_dict.keys():
            style = styles_dict[col]
            self.joint_graph_coverage_data[col].plot(style=style['style'],
                                                     lw=style['width'],
                                                     color=style['color'],
                                                     label=col)

        plt.xlabel('#Dependencies')
        plt.ylabel('#Solved problems')
        plt.grid(True)
        plt.yticks(range(0, 21, 5))
        plt.xlim(None, self.truncate_x_axis_at)

        plt.box(False)
        # plt.ylim(0, 21)
        if self.add_legend:
            plt.legend(loc='lower right')

        if self.save_figure:
            plt.savefig(fr'{output_folder}\coverage_{self.planner_type}_{self.domain_name}.png', dpi=100,
                        bbox_inches='tight')
        else:
            plt.show()

    def analyze_summarized_coverage_results(self):
        """ Summarizes the coverage results of all the solvers into one joint graph. Also adds the hindsight graph.
        """
        joint_graph_df = self.get_empty_df_graph()
        joint_graph_df['Hindsight'] = self.hindsight_graph_data['coverage']
        for solver_type, df_graph_solver in self.solver2coverage_graph_data.items():
            if self.has_random_solver and solver_type == self.random_solver_name:
                joint_graph_df[solver_type] = df_graph_solver['coverage']
                joint_graph_df['Random_upper_bound'] = df_graph_solver['upper_bound']
                joint_graph_df['Random_lower_bound'] = df_graph_solver['lower_bound']
            else:
                joint_graph_df[solver_type] = df_graph_solver
        self.joint_graph_coverage_data = joint_graph_df

    def calculate_hindsight_graph(self):
        """ Calculates the hindsight graph DataFrame, and save it to self.hindsight_graph_data
        """
        df_graph = self.get_empty_df_graph()
        df_graph['coverage'] = 0
        for problem in self.problems:
            hindsight = self.problem2hindsight_val[problem]
            for i, row in df_graph.iterrows():
                if i >= hindsight:
                    row['coverage'] += 1
        self.hindsight_graph_data = df_graph

    def analyze_coverage_for_solver(self, solver_type, random_solver):
        """ Analyze the coverage of a specific solver

        :param solver_type: the solver type we want to analyze
        :type solver_type: str
        :param random_solver: indicates whether this solver is a random solver (True) or not (False)
        :type random_solver: bool
        """
        df_graph = self.get_empty_df_graph()
        df_graph['coverage'] = float(0)
        if random_solver:
            df_graph['lower_bound'] = float(0)
            df_graph['upper_bound'] = float(0)
        for problem in self.problems:
            df = self.solver2dict_problem2data[solver_type][problem]

            previous_val = 0
            previous_upper_val = 0
            previous_lower_val = 0
            for i, row in df_graph.iterrows():
                if i in df[' amount of dependecies published'].values:
                    sub_df = df.loc[df[' amount of dependecies published'] == i]
                    if random_solver:
                        trials = sub_df[' success/failure'] == '  success'
                        success_rate = np.average(trials)
                        row['coverage'] += success_rate
                        previous_val = success_rate

                        previous_upper_val = int(np.any(trials))
                        previous_lower_val = int(np.all(trials))
                        row['lower_bound'] += previous_lower_val
                        row['upper_bound'] += previous_upper_val
                    else:
                        if sub_df[' success/failure'].iloc[0] == '  success':
                            row['coverage'] += 1
                            previous_val = 1
                        else:
                            previous_val = 0
                else:
                    # if the current amount of dependencies is never revealed in the current problem,
                    # then just assume it is as it was when the problem did reveal this amount of dependencies.
                    row['coverage'] += previous_val
                    if random_solver:
                        row['lower_bound'] += previous_lower_val
                        row['upper_bound'] += previous_upper_val
        self.solver2coverage_graph_data[solver_type] = df_graph

    def get_empty_df_graph(self):
        """ Creates an empty DataFrame with the index set to be the x-axis of the domain.

        :return: the new df with the index set to be the x-axis of the domain
        :rtype: pd.DataFrame
        """
        xaxis_set = self.published_dep_axis_set
        df_graph = pd.DataFrame()
        df_graph['published_dep'] = list(xaxis_set)
        df_graph.set_index('published_dep', inplace=True)
        df_graph.sort_index(inplace=True)
        return df_graph

    def analyze_cost(self):
        """ Analyzes the cost features in the results file.
        """
        for solver_type in self.solver2data.keys():
            if self.has_random_solver and solver_type == self.random_solver_name:
                # Do not add the random solver to the cost table
                continue
            self.analyze_cost_table(solver_type)

    def analyze_cost_table(self, solver_type):
        """ Analyzes the cost table entry of the given solver.

        :param solver_type: the solver we want to analyze it's cost table entry
        :type solver_type: str
        """
        min_costs = []
        max_costs = []
        min_dep_costs = []
        max_dep_costs = []
        improvements = []
        for problem in self.problems:
            success_df = self.solver2dict_problem2success_data[solver_type][problem]
            if not success_df.empty:
                # There was a success
                cost_col = success_df[' plan make span']

                min_cost = cost_col.min()
                max_cost = cost_col.max()
                min_dep_cost = cost_col.iloc[0]
                max_dep_cost = cost_col.iloc[-1]
                improvement = (min_dep_cost - min_cost) / min_dep_cost

                min_costs.append(min_cost)
                max_costs.append(max_cost)
                min_dep_costs.append(min_dep_cost)
                max_dep_costs.append(max_dep_cost)
                improvements.append(improvement)
            else:
                print(f'problem: {problem} was never solved')

        self.cost_table_entry[solver_type] = {'Min': np.mean(min_costs),
                                              'Max': np.mean(max_costs),
                                              'Min. dep': np.mean(min_dep_costs),
                                              'Max. dep': np.mean(max_dep_costs),
                                              'Imp.': np.mean(improvements)}

    def initialize_percentages(self):
        """ Initializes the self.percentages list, with all of the percentages we need to reveal.
            The list shall contain the following percentages: [0:0.05:1]
        """
        self.percentages = list(drange(0, 1, '0.05'))

    def find_first_solved_for_each_problem(self, solver_type):
        """ Finds the dependencies needed for solving each problem for the first time.

        :param solver_type: the solver we are checking now
        :type solver_type: str
        """
        prob2success_df = self.solver2dict_problem2success_data[solver_type]
        problem2first_solved = {}
        for prob in self.problems:
            success_df = prob2success_df[prob]
            if not success_df.empty:
                dep_first_solved = success_df[' amount of dependecies published'].iloc[0]
            else:
                dep_first_solved = -1
            problem2first_solved[prob] = dep_first_solved
        self.solver2dict_problem2first_solved[solver_type] = problem2first_solved


# Usage example:
# analyzer = DomainAnalyzer(planner_type='Joint_Projection', domain_name='BlocksWorld', save_figure=False)
# for i in range(1, 5):
#     analyzer.add_solver('m' + str(i), r"C:\Users\User\Desktop\second_degree\תזה\ICAPS2021\results_analyzer\m" + str(i) + r".csv")
# analyzer.analyze_results(r'result\Joint_Projection\figures')
# i=0
