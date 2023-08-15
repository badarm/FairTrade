import torch
from torch import nn
from torch.nn import functional as F


class ConstraintLoss(nn.Module):
    def __init__(self, n_class=2, alpha=1, p_norm=2):
        super(ConstraintLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.p_norm = p_norm
        self.n_class = n_class
        self.n_constraints = 2
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X=None, y=None, sensitive=None):
        return torch.zeros(self.n_constraints)

    def forward(self, X, out, sensitive, y=None):
        sensitive = sensitive.view(out.shape)
        if isinstance(y, torch.Tensor):
            y = y.view(out.shape)
        out = torch.sigmoid(out)
        mu = self.mu_f(X=X, out=out, sensitive=sensitive, y=y)
        gap_constraint = F.relu(
            torch.mv(self.M.to(self.device), mu.to(self.device)) - self.c.to(self.device)
        )
        if self.p_norm == 2:
            cons = self.alpha * torch.dot(gap_constraint, gap_constraint)
        else:
            cons = self.alpha * torch.dot(gap_constraint.detach(), gap_constraint)
        return cons
'''
#sandipan's suggestion
class AverageTreatmentEffectLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        """loss of average treatment effect

        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = sensitive_classes
        self.n_class = len(sensitive_classes)
        super(AverageTreatmentEffectLoss, self).__init__(
            n_class=self.n_class, alpha=alpha, p_norm=p_norm
        )
        self.n_constraints = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        for i in range(self.n_constraints):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X, out, sensitive, y=None):
        expected_values_list = []
        for v in self.sensitive_classes:
            idx_true = sensitive == v  # torch.bool
            expected_values_list.append(y[idx_true].mean())
        expected_values_list.append(y.mean())
        return torch.stack(expected_values_list)
    
    def forward(self, X, out, sensitive, y):
        return super(AverageTreatmentEffectLoss, self).forward(X, out, sensitive,y=y)
'''
class AverageTreatmentEffectLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        """loss of average treatment effect

        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = sensitive_classes
        self.n_class = len(sensitive_classes)
        super(AverageTreatmentEffectLoss, self).__init__(
            n_class=self.n_class, alpha=alpha, p_norm=p_norm
        )
        self.n_constraints = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        for i in range(self.n_constraints):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X, out, sensitive, y=None):
        
        expected_values_list = []
        tp_protected = 0.
        tn_protected = 0.
        fp_protected = 0.
        fn_protected = 0.

        tp_non_protected = 0.
        tn_non_protected = 0.
        fp_non_protected = 0.
        fn_non_protected = 0.
        
        saValue = 0
        for idx in range(len(sensitive)):
            # protrcted population
            if sensitive[idx] == saValue:

                # correctly classified
                if y[idx] == out[idx]:
                    if y[idx] == 1:
                        tp_protected += 1.
                    else:
                        tn_protected += 1.
      
                else:
                    if y[idx] == 1:
                        fn_protected += 1.
                    else:
                        fp_protected += 1.

            else:

                # correctly classified
                if y[idx] == out[idx]:
                    if y[idx] == 1:
                        tp_non_protected += 1.
                    else:
                        tn_non_protected += 1.
                # misclassified
                else:
                    if y[idx] == 1:
                        fn_non_protected += 1.
                    else:
                        fp_non_protected += 1.

        if((tp_protected + fn_protected)==0):
            tpr_protected = 0
        else:
            tpr_protected = tp_protected / (tp_protected + fn_protected)
        #tnr_protected = tn_protected / (tn_protected + fp_protected)

        if((tp_non_protected + fn_non_protected) == 0):
            tpr_non_protected=0
        else:
            tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
        #tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)


        #eqop = tpr_non_protected - tpr_protected
        expected_values_list.append(torch.tensor(tpr_non_protected, dtype=torch.float32))
        expected_values_list.append(torch.tensor(tpr_protected, dtype=torch.float32))
        expected_values_list.append(torch.tensor(tpr_protected, dtype=torch.float32))
        
        '''
        diff = out-y
        left = sensitive*diff #left is corresponding to non-protected value of gender
        left = left.mean(dim=0)
        expected_values_list.append(left.squeeze())
        right = (1-sensitive)*diff #right is corresponding to protected value of gender
        right = right.mean(dim=0)
        expected_values_list.append(right.squeeze())
        expected_values_list.append(right.squeeze())
        '''
        return torch.stack(expected_values_list)
    
    def forward(self, X, out, sensitive, y):
        return super(AverageTreatmentEffectLoss, self).forward(X, out, sensitive,y=y)

class DemographicParityLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        """loss of demograpfhic parity

        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = sensitive_classes
        self.n_class = len(sensitive_classes)
        super(DemographicParityLoss, self).__init__(
            n_class=self.n_class, alpha=alpha, p_norm=p_norm
        )
        self.n_constraints = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        for i in range(self.n_constraints):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X, out, sensitive, y=None):
        expected_values_list = []
        for v in self.sensitive_classes:
            idx_true = sensitive == v  # torch.bool
            expected_values_list.append(out[idx_true].mean())
            #("bismillah")
            
        expected_values_list.append(out.mean())
        return torch.stack(expected_values_list)

    def forward(self, X, out, sensitive, y=None):
        return super(DemographicParityLoss, self).forward(X, out, sensitive)


class EqualiedOddsLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        """loss of demograpfhic parity

        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = sensitive_classes
        self.y_classes = [0, 1]
        self.n_class = len(sensitive_classes)
        self.n_y_class = len(self.y_classes)
        super(EqualiedOddsLoss, self).__init__(n_class=self.n_class, alpha=alpha, p_norm=p_norm)
        # K:  number of constraint : (|A| x |Y| x {+, -})
        self.n_constraints = self.n_class * self.n_y_class * 2
        # J : dim of conditions  : ((|A|+1) x |Y|)
        self.dim_condition = self.n_y_class * (self.n_class + 1)
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        # make M (K * J): (|A| x |Y| x {+, -})  *   (|A|+1) x |Y|) )
        self.c = torch.zeros(self.n_constraints)
        element_K_A = self.sensitive_classes + [None]
        for i_a, a_0 in enumerate(self.sensitive_classes):
            for i_y, y_0 in enumerate(self.y_classes):
                for i_s, s in enumerate([-1, 1]):
                    for j_y, y_1 in enumerate(self.y_classes):
                        for j_a, a_1 in enumerate(element_K_A):
                            i = i_a * (2 * self.n_y_class) + i_y * 2 + i_s
                            j = j_y + self.n_y_class * j_a
                            self.M[i, j] = self.__element_M(a_0, a_1, y_1, y_1, s)

    def __element_M(self, a0, a1, y0, y1, s):
        if a0 is None or a1 is None:
            x = y0 == y1
            return -1 * s * x
        else:
            x = (a0 == a1) & (y0 == y1)
            return s * float(x)

    def mu_f(self, X, out, sensitive, y):
        expected_values_list = []
        for u in self.sensitive_classes:
            for v in self.y_classes:
                idx_true = (y == v) * (sensitive == u)  # torch.bool
                expected_values_list.append(out[idx_true].mean())
        # sensitive is star
        for v in self.y_classes:
            idx_true = y == v
            expected_values_list.append(out[idx_true].mean())
        return torch.stack(expected_values_list)

    def forward(self, X, out, sensitive, y):
        return super(EqualiedOddsLoss, self).forward(X, out, sensitive, y=y)
    
class EqualOpportunityLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        self.sensitive_classes = sensitive_classes
        self.y_classes = [1]  # only consider positive outcome
        self.n_class = len(sensitive_classes)
        self.n_y_class = len(self.y_classes)
        super(EqualOpportunityLoss, self).__init__(n_class=self.n_class, alpha=alpha, p_norm=p_norm)
        self.n_constraints = self.n_class * self.n_y_class * 2
        self.dim_condition = self.n_y_class * (self.n_class + 1)
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.c = torch.zeros(self.n_constraints)
        element_K_A = self.sensitive_classes + [None]
        for i_a, a_0 in enumerate(self.sensitive_classes):
            for i_y, y_0 in enumerate(self.y_classes):
                for i_s, s in enumerate([-1, 1]):
                    for j_y, y_1 in enumerate(self.y_classes):
                        for j_a, a_1 in enumerate(element_K_A):
                            i = i_a * (2 * self.n_y_class) + i_y * 2 + i_s
                            j = j_y + self.n_y_class * j_a
                            self.M[i, j] = self.__element_M(a_0, a_1, y_1, y_1, s)

    def __element_M(self, a0, a1, y0, y1, s):
        if a0 is None or a1 is None:
            x = y0 == y1
            return -1 * s * x
        else:
            x = (a0 == a1) & (y0 == y1)
            return s * float(x)

    def mu_f(self, X, out, sensitive, y):
        expected_values_list = []
        for u in self.sensitive_classes:
            for v in self.y_classes:
                idx_true = (y == v) * (sensitive == u)
                expected_values_list.append(out[idx_true].mean())
        # sensitive is star
        for v in self.y_classes:
            idx_true = y == v
            expected_values_list.append(out[idx_true].mean())
        return torch.stack(expected_values_list)

    def forward(self, X, out, sensitive, y):
        return super(EqualOpportunityLoss, self).forward(X, out, sensitive, y=y)
