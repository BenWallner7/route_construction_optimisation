# Route Construction Optimisation

You are tasked with solving a route construction optimization problem. Given three sets of 2D coordinate points, your objective is to construct an optimal path that maximizes the number of goal points visited while satisfying specific constraints.

## Different Optimisation Approaches

- Greedy Nearest-Neighbour algorithm with constraint validation.

## Key scripts

- `solve.py`: it is the main solver script. It runs one method at a time and produces a JSON solution file.

Example usage: `python solve.py --input instances.json --output solution.json --method greedy`

- `evaluate_methods.py`: runs multiple algorithms (greedy, beam, annealing, local, hybrid) on the full dataset and collects performance metrics for comparison. Results are written to a CSV file.

### Example usages

- Evaluate all methods on all instances: `python evaluate_methods.py --input instances.json --output results.csv`

- Evaluate only greedy and hybrid methods: `python evaluate_methods.py --input instances.json --output results_subset.csv --methods greedy hybrid`

- Evaluate all methods with 5s timeout and verbose logging: `python evaluate_methods.py --input instances.json --output results_verbose.csv --timeout 5 --verbose`

## Arguments

--input (required): path to instances.json

--output (required): path to write solution.json

--method (required): choose from:

    greedy – nearest-neighbor greedy heuristic

    beam – beam search with limited width

    annealing – simulated annealing metaheuristic

    local – greedy + local 2-opt refinement

    hybrid – greedy followed by 2-opt optimization

--verbose (optional): print detailed logs per instance

--timeout (optional): per-instance timeout (seconds)

## Outputs

- `solution.json`: solutions for the input instance file

## Analysis Scripts

The `./method_analysis/` directory contains an 'evaluate_methods.py' script which will run each of the 5 methods considered and record their outcomes to a csv file, this is then fed into the 'method_comparison_analysis.py' file which will compare the 5 methods with key numerical analysis.

'route_opt_results.csv' is the file produced by 'evaluate_methods.py' which contains the key results for each method.

'analysis_report.txt' is the report produced from 'method_comparison_analysis.py'  which compares the different methods based on various benchmarks.

## Past solutions

The `./initial_solution/` directory contains the first solution considered which is the standard greedy nearest neighbour approach. 