from domain_analyzer import DomainAnalyzer
from planner_analyzer import PlannerAnalyzer

# ********************************************************************************************************************
# change the base path in your machine:
# base_folder_name = r'C:\Users\User\Desktop\second_degree\code\GPPP(last_v)'  # my computer path
base_folder_name = r'D:\GPPP(last_v)'  # left server path
# base_folder_name = r'D:\rotem\GPPP(last_v)'  # right server path
# ********************************************************************************************************************
# choose the domains, planners and solvers that you want to analyze their results now:
specific_domains = ['BlocksWorld', 'Elevators']
domains_to_draw_graphs_to = {'BlocksWorld', 'Elevators'}
specific_planners = ['Joint_Projection']
specific_solvers = ['m1', 'm2', 'm3', 'm4', 'Random']
# ********************************************************************************************************************

all_domains = {'BlocksWorld': 'blocksworld',
               'Depot': 'depot',
               'Driverlog': 'driverlog',
               'Elevators': 'elevators08',
               'Logistics': 'logistics00',
               'Rovers': 'rovers',
               'ZenoTravel': 'zenotravel'}

all_planners = {'MAFS': 'MAFS_Projection_IJCAI',
                'Joint_Projection': 'Projection_Only_IJCAI'}

all_solvers = {'m1': 'Actions_Achiever',
               'm2': 'Public_Predicates_Achiever',
               'm3': 'New_Actions_Achiever',
               'm4': 'New_Public_Predicates_Achiever',
               'Random': 'Random'}

main_results_path = base_folder_name + r'\Experiment'
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
            if s == 'Random':
                random_solver = True
            else:
                random_solver = False
            solver_path = planner_path + '\\' + all_solvers[s] + '\\' + all_domains[d]
            experiments_results_file_path = solver_path + r'\Experiment_Output_File\output.csv'
            domain_analyzer.add_solver(s, experiments_results_file_path, random_solver)
        planner_analyzer.add_domain_analyzer(domain_analyzer)
    planner_analyzer.analyze_planner_results(domains_to_draw_graphs_to)
