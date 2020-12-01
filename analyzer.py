import pandas as pd
import decimal


def drange(x, y, jump):
  while x <= y:
    yield float(x)
    x += decimal.Decimal(jump)


class DomainAnalyzer:


    def __init__(self, planner_type, domain_name):
        """ Initializes a new DomainAnalyzer instance

        :param planner_type: the type of the planner we are analyzing now. Can be 'MAFS', or 'Joint Projection'
        :type planner_type: str
        :param domain_name: the name of the domain we are analyzing now
        :type domain_name: str
        """
        self.planner_type = planner_type
        self.domain_name = domain_name
        self.solver2csv_file = {}  # {solver_type: csv_file_path}
        self.solver2data = {}  # {solver_type: data_for_solver(DataFrame)}
        self.problems = None  # set of all of the problem names of this domain
        self.solver2dict_problem2data = {}  # {solver_type: {problem_name: data_of_problem(DataFrame)}}
        self.percentages = None  # all of the percentages of revealing dependencies.
        self.initialize_percentages()

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
        all_names = set(df[' folder name'].unique())
        if self.problems is None:
            self.problems = all_names
        elif self.problems != all_names:
            raise Exception('The problems must be the same in all of the solvers!')

        problem2data = {}
        for problem in self.problems:
            sub_df = df[df[' folder name'] == problem]
            sub_df.sort_values(by=['Percentage of actions selected'], inplace=True)
            problem2data[problem] = sub_df
            all_percentages = list(sub_df['Percentage of actions selected'])
            if all_percentages != self.percentages:
                raise Exception('The percentages must agree!')

        self.solver2dict_problem2data[solver_type] = problem2data

    def analyze_coverage(self, output_folder_path):
        """ Analyze the coverage of the results files for each of the solvers.

        :param output_folder_path: the path to save the results at
        :type output_folder_path: str
        """
        for solver_type in self.solver2data.keys():
            self.analyze_coverage_for_solver(solver_type)
        self.analyze_summarized_coverage_results()

    def analyze_coverage_for_solver(self, solver_type):
        """ Analyze the coverage of a specific solver

        :param solver_type: the solver type we want to analyze
        :type solver_type: str
        """
        raise Exception('Need to implement')

    def initialize_percentages(self):
        """ Initializes the self.percentages list, with all of the percentages we need to reveal.
            The list shall contain the following percentages: [0:0.05:1]
        """
        self.percentages = list(drange(0, 1, '0.05'))


analyzer = DomainAnalyzer(planner_type='MAFS', domain_name='elevators')
analyzer.add_solver('m1', r"C:\Users\User\Desktop\second_degree\code\GPPP(last_v)\Experiment\MAFS_results_from_server\second exp\Actions_Achiever\elevators08\Experiment_Output_File\output.csv")
analyzer.analyze_results('output')
