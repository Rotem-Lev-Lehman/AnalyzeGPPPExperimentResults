from domain_analyzer import DomainAnalyzer
from pathlib import Path


class PlannerAnalyzer:

    def __init__(self, planner_type):
        """ Initializes a new PlannerAnalyzer instance

        :param planner_type: the type of the planner we are analyzing now. Can be 'MAFS', or 'Joint_Projection'
        :type planner_type: str
        """
        self.planner_type = planner_type
        self.domain_analyzers = []  # this will contain all of the DomainAnalyzer objects.
        self.coverage_table = None  # this will contain the planner's coverage table.
        self.cost_table = None  # this will contain the planner's cost table.
        self.solvers = set()  # this will contain all of the solvers we are using.
        # results saving path:
        self.results_folder_path = fr'results\{self.planner_type}'
        Path(self.results_folder_path).mkdir(parents=True, exist_ok=True)
        self.graphs_folder_path = self.results_folder_path + r'\figures'
        Path(self.graphs_folder_path).mkdir(parents=True, exist_ok=True)

    def add_domain_analyzer(self, domain_analyzer):
        """ Adds the domain analyzer to the planner analyzer.

        :param domain_analyzer: the DomainAnalyzer object we want to add to this planner analyzer
        :type domain_analyzer: DomainAnalyzer
        """
        self.domain_analyzers.append(domain_analyzer)

    def analyze_planner_results(self):
        """ Analyzes the planner results for all of the given domain analyzers.
        """
        self.generate_tables()
        self.write_tables_to_csv_file()

    def generate_tables(self):
        """ Generates the results tables, and saves graphs.
        """
        coverage_table = {}
        cost_table = {}
        for domain_analyzer in self.domain_analyzers:
            domain_analyzer.analyze_results(self.graphs_folder_path)

            coverage_table_entry = domain_analyzer.coverage_table
            max_dep = domain_analyzer.max_dep
            cost_table_entry = domain_analyzer.cost_table_entry

            coverage_table[domain_analyzer.domain_name] = {'max_dep': max_dep,
                                                           'coverage_table_entry': coverage_table_entry}
            cost_table[domain_analyzer.domain_name] = cost_table_entry

            for solver in cost_table_entry.keys():
                self.solvers.add(solver)

        self.coverage_table = coverage_table
        self.cost_table = cost_table

    def write_tables_to_csv_file(self):
        """ Writes the results tables to csv files.
        """
        self.write_coverage_table_to_csv()
        self.write_cost_table_to_csv()

    def write_coverage_table_to_csv(self):
        """ Writes the coverage results table to a csv file.
        """
        file_path = self.results_folder_path + r'\coverage_table.csv'
        with open(file_path, 'w') as file1:
            # first write the header lines to the file:
            header1_line = f'Domain,#dep,{self.planner_type}'
            # the beginning of the following rows can be removed, as it will be merged in the excel file:
            header2_line = ','
            header3_line = ','
            sorted_solvers = list(sorted(self.solvers))
            for solver in sorted_solvers:
                header2_line += f',{solver},'
                header3_line += ',#C,M_D'

            file1.write(f'{header1_line}\n')
            file1.write(f'{header2_line}\n')
            file1.write(f'{header3_line}\n')

            # now lets write the data to the file:
            for domain in self.coverage_table.keys():
                entry = self.coverage_table[domain]
                max_dep = entry['max_dep']
                coverage_entry = entry['coverage_table_entry']
                curr_line = f'{domain},{max_dep}'
                for solver in sorted_solvers:
                    solver_entry = coverage_entry[solver]
                    C = solver_entry['C']
                    Md = solver_entry['Md']
                    curr_line += f',{C},{Md}'
                file1.write(f'{curr_line}\n')

    def write_cost_table_to_csv(self):
        """ Writes the cost results table to a csv file.
        """
        file_path = self.results_folder_path + r'\cost_table.csv'
        with open(file_path, 'w') as file1:
            # first write the header lines to the file:
            header1_line = f'Domain,M,{self.planner_type}'
            # the beginning of the following rows can be removed, as it will be merged in the excel file:
            header2_line = ','
            sorted_solvers = list(sorted(self.solvers))
            columns = ['Min', 'Max', 'Min. dep', 'Max. dep', 'Imp.']
            for col in columns:
                header2_line += f',{col}'
            file1.write(f'{header1_line}\n')
            file1.write(f'{header2_line}\n')

            # now lets write the data to the file:
            for domain in self.cost_table.keys():
                cost_entry = self.cost_table[domain]
                for solver in sorted_solvers:
                    solver_entry = cost_entry[solver]
                    curr_line = f'{domain},{solver}'
                    for col in columns:
                        curr_line += f',{solver_entry[col]}'
                    file1.write(f'{curr_line}\n')


# Usage example:
# domain_analyzer = DomainAnalyzer(planner_type='Joint_Projection', domain_name='BlocksWorld')
# for i in range(1, 5):
#     domain_analyzer.add_solver('m' + str(i), r"C:\Users\User\Desktop\second_degree\תזה\ICAPS2021\results_analyzer\m" + str(i) + r".csv")
# planner_analyzer = PlannerAnalyzer(planner_type='Joint_Projection')
# planner_analyzer.add_domain_analyzer(domain_analyzer)
# planner_analyzer.analyze_planner_results()
