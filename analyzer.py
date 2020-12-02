import pandas as pd
import decimal
import matplotlib.pyplot as plt


def drange(x, y, jump):
  while x <= y:
    yield float(x)
    x += decimal.Decimal(jump)


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

    def add_solver(self, solver_type, csv_file_path):
        """ Adds a solver to the DomainAnalyzer.

        :param solver_type: the type of the solver
        :type solver_type: str
        :param csv_file_path: the path to the results of the solver
        :type csv_file_path: str
        """
        self.solver2csv_file[solver_type] = csv_file_path

    def analyze_results(self, output_folder_path):
        """ Analyzes the results given, and creates graphs and summarized DataFrames.

        :param output_folder_path: the path to save the results at
        :type output_folder_path: str
        """
        self.read_results()
        self.analyze_coverage(output_folder_path)
        self.analyze_cost(output_folder_path)

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
            all_percentages = list(sub_df['Percentage of actions selected'])
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
            elif self.problem2hindsight_val[problem] > hindsight:
                self.problem2hindsight_val[problem] = hindsight

        self.solver2dict_problem2data[solver_type] = problem2data
        self.solver2dict_problem2success_data[solver_type] = problem2success_data

    def analyze_coverage(self, output_folder_path):
        """ Analyze the coverage of the results files for each of the solvers.

        :param output_folder_path: the path to save the results at
        :type output_folder_path: str
        """
        for solver_type in self.solver2data.keys():
            self.analyze_coverage_for_solver(solver_type)
        self.calculate_hindsight_graph()
        self.analyze_summarized_coverage_results()
        self.show_coverage_graph()
        i=0

    def show_coverage_graph(self):
        width1 = 4
        height1 = 2
        width_height_1 = (width1, height1)

        plt.figure(figsize=width_height_1)

        if not self.same_graph_line:
            styles_dict = {'m1': {'style': '-', 'color': 'black', 'width': 3},
                           'm2': {'style': '-.', 'color': 'red', 'width': 3},
                           'm3': {'style': '-', 'color': 'green', 'width': 2},
                           'm4': {'style': '--', 'color': 'purple', 'width': 2},
                           'Hindsight': {'style': '--', 'color': 'yellow', 'width': 2}
                           }
        else:
            styles_dict = {'m1': {'style': '-', 'color': 'blue', 'width': 2},
                           'Hindsight': {'style': '--', 'color': 'yellow', 'width': 2}
                           }

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
            plt.savefig(fr'figures\figure_{self.planner_type}_{self.domain_name}.png', dpi=100, bbox_inches='tight')
        else:
            plt.show()

    def analyze_summarized_coverage_results(self):
        """ Summarizes the coverage results of all the solvers into one joint graph. Also adds the hindsight graph.
        """
        joint_graph_df = self.get_empty_df_graph()
        joint_graph_df['Hindsight'] = self.hindsight_graph_data['coverage']
        for solver_type, df_graph_solver in self.solver2coverage_graph_data.items():
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

    def analyze_coverage_for_solver(self, solver_type):
        """ Analyze the coverage of a specific solver

        :param solver_type: the solver type we want to analyze
        :type solver_type: str
        """
        df_graph = self.get_empty_df_graph()
        df_graph['coverage'] = 0
        for problem in self.problems:
            df = self.solver2dict_problem2data[solver_type][problem]
            previously_success = False
            for i, row in df_graph.iterrows():
                if i in df[' amount of dependecies published'].values:
                    sub_df = df.loc[df[' amount of dependecies published'] == i]
                    if sub_df[' success/failure'].iloc[0] == '  success':
                        row['coverage'] += 1
                        previously_success = True
                    else:
                        previously_success = False
                elif previously_success:
                    # if the current amount of dependencies is never revealed in the current problem,
                    # then just assume it is as it was when the problem did reveal this amount of dependencies.
                    row['coverage'] += 1
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

    def initialize_percentages(self):
        """ Initializes the self.percentages list, with all of the percentages we need to reveal.
            The list shall contain the following percentages: [0:0.05:1]
        """
        self.percentages = list(drange(0, 1, '0.05'))


analyzer = DomainAnalyzer(planner_type='Joint_Projection', domain_name='BlocksWorld')
for i in range(1, 5):
    analyzer.add_solver('m' + str(i), r"C:\Users\User\Desktop\second_degree\תזה\ICAPS2021\results_analyzer\m" + str(i) + r".csv")
analyzer.analyze_results('output')
