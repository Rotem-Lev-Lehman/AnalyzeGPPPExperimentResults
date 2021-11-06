import os
import pandas as pd
from pathlib import Path
import re

planners = ['Optimal', 'MAFS', 'Projection']
regular_selectors = ['Actions_Achiever', 'Public_Predicates_Achiever', 'New_Actions_Achiever', 'New_Public_Predicates_Achiever']
optimal_selectors = ['FF_and_SymPA']
selectors = {
    'Optimal': optimal_selectors,
    'MAFS': regular_selectors,
    'Projection': regular_selectors
}

interesting_columns_optimal = [' folder name', ' success/failure', ' amount of dependencies used', ' lower bound (for optimal dependencies)', ' upper bound (for optimal dependencies)']
interesting_columns_mafs = ['Percentage of actions selected', ' folder name', ' success/failure', ' amount of dependencies used', ' amount of dependecies published']
interesting_columns_projection = ['Percentage of actions selected', ' folder name', ' success/failure', ' amount of dependencies used', ' amount of dependecies published']

keep_columns = {
    'Optimal': interesting_columns_optimal,
    'MAFS': interesting_columns_mafs,
    'Projection': interesting_columns_projection
}

domains = ['blocksworld', 'depot', 'driverlog', 'elevators08', 'logistics00', 'rovers', 'zenotravel']

main_path = r'C:\Users\User\Desktop\second_degree\תזה\JAAMAS journal paper\ExperimentsRawResults'
path2results_file = r'Experiment_Output_File\output.csv'
results_folder = 'optimal_results_3'


def get_problem_number(filepath):
    match_res = re.match(r'\D+(\d+(-\d+(-\D+)?)?)', filepath.split('/')[1])
    return ' ' + match_res.group(1)


def get_first_solve(df):
    success = '  success'
    df_success = df[df[' success/failure'] == success]
    df_first_solve = df_success.sort_values('Percentage of actions selected', ascending=True).drop_duplicates(' folder name')
    return df_first_solve


def get_hindsight(df):
    success = '  success'
    df_success = df[df[' success/failure'] == success]
    df_hindsight = df_success.sort_values(' amount of dependencies used', ascending=True).drop_duplicates(' folder name')
    df_hindsight['hindsight'] = df_hindsight[' amount of dependencies used']
    df_hindsight = df_hindsight[['hindsight', 'problem_number']]
    return df_hindsight


def get_only_relevant_rows(df):
    idx_relevant = 'success' in df[' success/failure'] or df[' lower bound (for optimal dependencies)'] > 0
    return df[idx_relevant]


def split_problem_number(p):
    p = p.split(' ')[1]
    if '-' in p:
        p_split = p.split('-')
        p_0 = p_split[0]
        p_1 = p_split[1]
        if len(p_split) > 2:
            p_1 += '5'
    else:
        p_0 = p
        p_1 = '0'
    return p_0, p_1


def custom_problem_number(p):
    p_0, p_1 = split_problem_number(p)
    return float(f'{p_0}.{p_1}')


def update_hindsight(domain, problem, curr_hindsight):
    if domain not in domain2problem2hindsight:
        domain2problem2hindsight[domain] = {}
    if problem not in domain2problem2hindsight[domain]:
        domain2problem2hindsight[domain][problem] = curr_hindsight
    elif domain2problem2hindsight[domain][problem] > curr_hindsight:
        domain2problem2hindsight[domain][problem] = curr_hindsight


optimal_solved = {}
domain2problem2hindsight = {}

for p in planners:
    planner_path = os.path.join(main_path, p)
    planner_results = os.path.join(results_folder, p)
    Path(planner_results).mkdir(parents=True, exist_ok=True)
    for s in selectors[p]:
        selector_path = os.path.join(planner_path, s)
        selector_results = os.path.join(planner_results, f'{s}_results.csv')
        dfs_selector = []

        for d in domains:
            domain_path = os.path.join(selector_path, d, path2results_file)
            # domain_results = os.path.join(selector_results, f'{d}_results.csv')
            print(domain_path)

            df_domain = pd.read_csv(domain_path)
            # print(df_domain.columns)
            df_domain.drop_duplicates(subset=['Percentage of actions selected', ' folder name'], keep='first', inplace=True)
            df_domain = df_domain[keep_columns[p]]
            df_domain['problem_number'] = df_domain[' folder name'].apply(lambda name: get_problem_number(name))

            if p != 'Optimal':
                df_first_solve = get_first_solve(df_domain)
                df_hindsight = get_hindsight(df_domain)
                df_domain = pd.merge(df_first_solve, df_hindsight, on='problem_number')
                df_domain = df_domain[df_domain[' folder name'].apply(lambda name: name in optimal_solved[d])]
                for row in df_domain.iterrows():
                    update_hindsight(d, row[1]['problem_number'], row[1]['hindsight'])
                df_domain = df_domain[['Percentage of actions selected', ' folder name', 'problem_number', ' success/failure', ' amount of dependecies published']]
            else:
                df_domain = get_only_relevant_rows(df_domain)
                all_problems = list(df_domain[' folder name'])
                optimal_solved[d] = all_problems
            df_domain = df_domain.sort_values('problem_number', key=lambda x: x.map(custom_problem_number), ascending=True)
            # df_domain.to_csv(domain_results, index=False)
            dfs_selector.append(df_domain)
        df_selector = pd.concat(dfs_selector, axis=0)
        df_selector.to_csv(selector_results, index=False)


hindsight_filename = f'{results_folder}/hindsight_file.csv'
with open(hindsight_filename, mode='w') as hindsight_file:
    hindsight_file.write('Domain,Problem,Hindsight\n')
    for d in domains:
        domain_hindsight = dict(sorted(domain2problem2hindsight[d].items(), key=lambda item: custom_problem_number(item[0])))
        for p, h in domain_hindsight.items():
            hindsight_file.write(f'{d},{p},{h}\n')

print('done')
