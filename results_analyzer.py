from domain_analyzer import DomainAnalyzer
from planner_analyzer import PlannerAnalyzer

all_domains = {'BlocksWorld': 'blocksworld',
               'Depot': 'depot',
               'Driverlog': 'driverlog',
               'Elevators': 'elevators08',
               'Logistics': 'logistics00',
               'Rovers': 'rovers',
               'ZenoTravel': 'zenotravel'}

all_planners = {'MAFS': 'MAFS_Projection_ICAPS',
                'Joint_Projection': 'Projection_Only_ICAPS'}

all_solvers = {'m1': 'Actions_Achiever',
               'm2': 'Public_Predicates_Achiever',
               'm3': 'New_Actions_Achiever',
               'm4': 'New_Public_Predicates_Achiever'}

specific_domains = ['BlocksWorld']
specific_planners = ['Joint_Projection']
specific_solvers = ['m1', 'm2', 'm3', 'm4']

main_results_path = r'...Experiments'
print('Starting to analyze results.')
for p in specific_planners:
    planner_path = main_results_path + '\\' + all_planners[p] + r'\Dependecies\No_Collaboration'
    print('********************************************************************************************************')
    print(f'Now analyzing results of the planner: {p}')
    planner_analyzer = PlannerAnalyzer(p)
    for d in specific_domains:
        print(f'\tDomain: {d}')
        domain_analyzer = DomainAnalyzer(p, d)
        for s in specific_solvers:
            solver_path = planner_path + '\\' + all_solvers[s] + '\\' + all_domains[d]
            experiments_results_file_path = solver_path + r'\Experiment_Output_File\output.csv'
            domain_analyzer.add_solver(s, experiments_results_file_path)
        planner_analyzer.add_domain_analyzer(domain_analyzer)
    planner_analyzer.analyze_planner_results()
