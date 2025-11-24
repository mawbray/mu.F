
def post_process_setup(cfg, graph, model):
    post_process = graph.graph['post_process'](cfg, graph, model, 0)
    assert hasattr(post_process, 'run')
    # TODO: update this to allow flexibility for whether a sampling scheme or local SIP scheme is used
    post_process.load_training_methods(graph.graph["post_process_training_methods"])
    post_process.load_solver_methods(graph.graph["post_process_solver_methods"])
    post_process.graph.graph["solve_post_processing_problem"] = True
    return post_process

def post_process_sampling_setup(cfg, post_process, live_set, sampler):
    post_process.sampler = lambda : sampler()
    post_process.load_fresh_live_set(live_set=live_set(cfg, cfg.samplers.notion_of_feasibility))
    return post_process