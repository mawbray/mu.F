import ray
import casadi as ca
import jax.numpy as jnp 
ray.init()

# Define a function that solves an NLP problem, specifying it to run in a separate process
@ray.remote(num_cpus=1)
def solve_nlp(problem_id, problem_data):
    # Recreate the necessary non-serializable context here
    # For example, load data from a file or reinitialize an object

    # Define your NLP problem using CasADi and IPOPT
    x = ca.MX.sym('x', problem_data['n'])
    objective = ca.sumsqr(x)
    # Add your specific non-serializable constraints by recreating them
    cons_g = problem_data['cons_g']
    constraints = [cons_g(x)]

    nlp = {'x': x, 'f': objective, 'g': ca.vertcat(*constraints)}
    opts = {"ipopt.print_level": 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    sol = solver(x0 = [10]*problem_data['n'],lbg=0, ubg=0)
    
    return (solver.stats(), sol['x'])


if __name__ == "__main__":

    # Define your problems
    problems = [
        {'id': 1, 'data': {'n': 10, 'cons_g': lambda x: x[0] + x[1] - 10}},
        {'id': 2, 'data': {'n': 20, 'cons_g': lambda x: x[0] + x[1] - 20}},
        {'id': 3, 'data': {'n': 30, 'cons_g': lambda x: x[0] + x[1] - 20}},
        {'id': 4, 'data': {'n': 40, 'cons_g': lambda x: x[0] + x[1] - 20}},
        {'id': 5, 'data': {'n': 50, 'cons_g': lambda x: x[0] + x[1] - 20}},
        {'id': 6, 'data': {'n': 60, 'cons_g': lambda x: x[0] + x[1] - 20}},
        {'id': 7, 'data': {'n': 70, 'cons_g': lambda x: x[0] + x[1] - 20}},
        {'id': 8, 'data': {'n': 80, 'cons_g': lambda x: x[0] + x[1] - 20}},
        {'id': 9, 'data': {'n': 90, 'cons_g': lambda x: x[0] + x[1] - 20}},
        {'id': 10, 'data': {'n': 100, 'cons_g': lambda x: x[0] + x[1] - 20}},
        {'id': 11, 'data': {'n': 110, 'cons_g': lambda x: x[0] + x[1] - 20}},

        # Add more problems as needed
    ]

    # Execute the problems in parallel, ensuring each task runs in a separate process
    results = ray.get([solve_nlp.remote(problem['id'], problem['data']) for problem in problems])

    # Process the results
    for i, result in enumerate(results):
        print(f"Result for problem {problems[i]['id']}: {result}")

    ray.shutdown()

