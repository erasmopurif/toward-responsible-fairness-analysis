import numpy as np
import pandas as pd

class Fairness(object):
    """
    Compute fairness metrics
    """
    def __init__(self, df_profile, test_nodes_idx, targets, predictions, sens_attr, neptune_run,
            multiclass_pred=False, multiclass_sens=False):

        self.multiclass_pred = multiclass_pred
        self.multiclass_sens = multiclass_sens
        self.sens_attr = sens_attr
        self.neptune_run = neptune_run
        self.neptune_run["sens_attr"] = self.sens_attr
        self.df_profile = df_profile
        self.test_nodes_idx = test_nodes_idx.cpu().detach().numpy()
        self.true_y = targets # target variables
        self.pred_y = predictions # prediction of the classifier
        self.sens_attr_array = self.df_profile[self.sens_attr].values # sensitive attribute values
        self.sens_attr_values = self.sens_attr_array[self.test_nodes_idx]

        self.class_range = list(set(self.true_y))
        self.y_hat = []
        self.yneq_hat = []
        for y_hat_idx in self.class_range:
            self.y_hat.append(self.pred_y == y_hat_idx)
            self.yneq_hat.append(self.pred_y != y_hat_idx)
            
        self.sens_attr_range = list(set(self.sens_attr_values))
        self.s = []
        for s_idx in self.sens_attr_range:
            self.s.append(self.sens_attr_values == s_idx)

        self.y_s = []
        self.yneq_s = []
        for y_idx in self.class_range:
            self.y_s.append([])
            self.yneq_s.append([])
            for s_idx in self.sens_attr_range:
                self.y_s[y_idx].append(np.bitwise_and(self.true_y == y_idx, self.s[s_idx]))
                self.yneq_s[y_idx].append(np.bitwise_and(self.true_y != y_idx, self.s[s_idx]))
        self.y_s = np.array(self.y_s)
        self.yneq_s = np.array(self.yneq_s)

    
    def statistical_parity(self):
        """
        P(y^=0|s=0) = P(y^=0|s=1) = ... = P(y^=0|s=N)
        [...]
        P(y^=M|s=0) = P(y^=M|s=1) = ... = P(y^=M|s=N)
        """
        stat_parity = []
        for y_hat_idx in self.class_range:
            stat_parity.append([])
            for s_idx in self.sens_attr_range:
                stat_parity[y_hat_idx].append(
                    sum(np.bitwise_and(self.y_hat[y_hat_idx], self.s[s_idx])) /
                    sum(self.s[s_idx])
                )
                self.neptune_run["fairness/SP_y^" + str(y_hat_idx) + "_s" + str(s_idx)] = stat_parity[y_hat_idx][s_idx]

    
    def equal_opportunity(self):
        """
        P(y^=0|y=0,s=0) = P(y^=0|y=0,s=1) = ... = P(y^=0|y=0,s=N)
        [...]
        P(y^=M|y=M,s=0) = P(y^=M|y=M,s=1) = ... = P(y^=M|y=M,s=N)
        """
        equal_opp = []
        for y_hat_idx in self.class_range:
            equal_opp.append([])
            for s_idx in self.sens_attr_range:
                try:
                    equal_opp[y_hat_idx].append(
                        sum(np.bitwise_and(self.y_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) /
                        sum(self.y_s[y_hat_idx][s_idx])
                    )
                except ZeroDivisionError:
                    equal_opp[y_hat_idx].append(0)
                self.neptune_run["fairness/EO_y" + str(y_hat_idx) + "_s" + str(s_idx)] = equal_opp[y_hat_idx][s_idx]


    def overall_accuracy_equality(self):
        ''' P(y^=0|y=0,s=0) + ... + P(y^=M|y=M,s=0) = ... = P(y^=0|y=0,s=N) + ... + P(y^=M|y=M,s=N) '''
        oae_s = []
        for s_idx in self.sens_attr_range:
            oae_temp = 0.0
            for y_hat_idx in self.class_range:
                try:
                    oae_temp += (
                        sum(np.bitwise_and(self.y_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) /
                        sum(self.y_s[y_hat_idx][s_idx])
                    )
                except ZeroDivisionError:
                    oae_temp += 0.0
            oae_s.append(oae_temp)
            self.neptune_run["fairness/OAE_s" + str(s_idx)] = oae_s[s_idx]


    def treatment_equality(self):
        """
        P(y^=0|y/=0,s=0) / P(y^/=0|y=0,s=0) = ... = P(y^=0|y/=0,s=N) / P(y^/=0|y=0,s=N)
        [...]
        P(y^=M|y/=M,s=0) / P(y^/=M|y=M,s=0) = ... = P(y^=M|y/=M,s=N) / P(y^/M|y=M,s=N)
        """
        te_fp_fn = []
        te_fn_fp = []
        te = []
        for y_hat_idx in self.class_range:
            te_fp_fn.append([])
            te_fn_fp.append([])
            abs_te_fp_fn = 0.0
            abs_te_fn_fp = 0.0
            te.append([])
            for s_idx in self.sens_attr_range:
                try:
                    te_fp_fn[y_hat_idx].append(
                        (sum(np.bitwise_and(self.y_hat[y_hat_idx], self.yneq_s[y_hat_idx][s_idx])) / sum(self.yneq_s[y_hat_idx][s_idx])) /
                        (sum(np.bitwise_and(self.yneq_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) / sum(self.y_s[y_hat_idx][s_idx]))
                    )
                except ZeroDivisionError:
                    te_fp_fn[y_hat_idx].append(100)
                
                try:
                    te_fn_fp[y_hat_idx].append(
                        (sum(np.bitwise_and(self.yneq_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) / sum(self.y_s[y_hat_idx][s_idx])) /
                        (sum(np.bitwise_and(self.y_hat[y_hat_idx], self.yneq_s[y_hat_idx][s_idx])) / sum(self.yneq_s[y_hat_idx][s_idx]))
                    )
                except ZeroDivisionError:
                    te_fn_fp[y_hat_idx].append(100)

                abs_te_fp_fn += abs(te_fp_fn[y_hat_idx][s_idx])
                abs_te_fn_fp += abs(te_fn_fp[y_hat_idx][s_idx])
        
                if abs_te_fp_fn < abs_te_fn_fp:
                    te[y_hat_idx].append(te_fp_fn[y_hat_idx][s_idx])
                else:
                    te[y_hat_idx].append(te_fn_fp[y_hat_idx][s_idx])

        for y_idx in self.class_range:
            for s_idx in self.sens_attr_range:
                self.neptune_run["fairness/TE_y" + str(y_idx) + "_s" + str(s_idx)] = te[y_idx][s_idx]
