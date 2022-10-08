import torch
import numpy as np
import neptune.new as neptune

from parser import parameter_parser
from clustering import ClusteringMachine
from clustergnn import ClusterGNNTrainer

from utils import pos_preds_attr_distr, tab_printer, graph_reader, field_reader, target_reader, label_reader
import time

from fairness import Fairness

def main():
    start_time = time.perf_counter()

    """
    Parsing command line parameters, reading data, graph decomposition, fitting and scoring the model.
    """
    args = parameter_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    field_index = field_reader(args.field_path)
    target = target_reader(args.target_path)
    user_labels = label_reader(args.labels_path)

    # Instantiate Neptune client and log arguments
    neptune_run = neptune.init(
        project = args.neptune_project,
        api_token = args.neptune_token)
    neptune_run["sys/tags"].add(args.log_tags.split(","))
    neptune_run["seed"] = args.seed
    neptune_run["dataset"] = "JD-small" if "jd" in args.edge_path else "Alibaba-small"
    neptune_run["model"] = "CatGCN"
    neptune_run["label"] = args.label
    neptune_run["lr"] = args.learning_rate
    neptune_run["L2"] = args.weight_decay
    neptune_run["dropout"] = args.dropout
    neptune_run["diag_probe"] = args.diag_probe
    neptune_run["nfm_units"] = args.nfm_units
    neptune_run["grn_units"] = args.grn_units
    neptune_run["gnn_hops"] = args.gnn_hops
    neptune_run["gnn_units"] = args.gnn_units
    neptune_run["balance_ratio"] = args.balance_ratio
    # neptune_run["n_epochs"] = args.epochs

    clustering_machine = ClusteringMachine(args, graph, field_index, target)
    clustering_machine.decompose()
    gnn_trainer = ClusterGNNTrainer(args, clustering_machine, neptune_run)
    gnn_trainer.train_val_test()

    ## Compute accuracy per sensitive attribute group
    # pos_preds_distr = pos_preds_attr_distr(user_labels, gnn_trainer.targets, gnn_trainer.predictions, clustering_machine.sg_test_nodes[0], args.label, "age")
    # print(pos_preds_distr)
    # neptune_run["pos_preds_distr"] = pos_preds_distr

    ## Compute fairness metrics
    print("Fairness metrics on sensitive attributes '{}':".format(args.sens_attr))
    fair_obj = Fairness(user_labels, clustering_machine.sg_test_nodes[0], gnn_trainer.targets, gnn_trainer.predictions, args.sens_attr, neptune_run, args.multiclass_pred, args.multiclass_sens)
    fair_obj.statistical_parity()
    fair_obj.equal_opportunity()
    fair_obj.overall_accuracy_equality()
    fair_obj.treatment_equality()

    elaps_time = (time.perf_counter() - start_time)/60
    neptune_run["elaps_time"] = elaps_time

    neptune_run.stop()

if __name__ == "__main__":
    main()
