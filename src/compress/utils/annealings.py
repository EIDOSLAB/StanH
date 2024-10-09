import torch.nn as nn 
import math
import torch



class Annealings(nn.Module):
    def __init__( self, iteration = 1500, 
                        beta = 1, 
                        factor = 50, 
                        type = "gap", 
                        decreasing = False,
                        dec_epoch = -1, 
                        decreasing_factor = 0,
                        threshold = 0.02,
                        mode = "min",
                        threshold_mode = "abs",
                        patience = 10,
                        max_beta = 1e3,
                        starting_epochs = -1):






        super().__init__()

        self.starting_epochs = starting_epochs
        self.loss = []
        self.factor = factor 
        self.type = type
        self.iteration = iteration
        self.gap = 0
        self.beta = beta 
        self.factor = factor
        self.decreasing_factor = decreasing_factor
        self.decreasing = decreasing 
        self.dec_epoch = dec_epoch
        self.threshold = threshold

        self.mode = mode 
        self.threshold_mode = threshold_mode
        self.patience = patience
        self.max_beta = max_beta


        self.num_bad_epochs = None
        self.best = 1e2 # very high number 
        self.list_epoch = [] 
        self.current_epoch = 0
        self.counter = 0
        self.beta_list = [self.beta]
        self.beta_max = self.beta


        assert self.type in ("linear_stoc","linear","gap","constant","loss","AugmentBetaOnPlateau","gap_stoc")
        
    def update_gap(self,gp):
        self.gap = gp

    def update_loss(self, cl):
        self.loss.append(cl)




    def is_better(self,a, best):
        if self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1. - self.threshold
            return a < best*rel_epsilon
        if self.mode == "max" and self.threshold_mode == "abs":
            return a > best - self.threshold
        else:
            rel_epsilon = 1. - self.threshold
            return a > best*rel_epsilon
    

    def update_max_beta(self):
        self.max_beta = 1
        self.beta = 1

    def step(self, gap, epoch, lss, plat = False):
        if self.type == "linear":
            if  self.beta < 50000:
                if self.decreasing is False or self.dec_epoc > epoch:
                    self.beta += self.factor/self.iteration 
                else:
                    self.beta -= self.decreasing_factor/self.iteration  
            else:
                self.beta = self.beta/2
            
        elif self.type == "linear_stoc":
            self.max_beta += self.factor/self.iteration
            self.beta = torch.empty_like(torch.tensor([0.0])).uniform_(1,  self.beta_max).item() 
        
        
        
        elif self.type == "gap":
            self.update_gap(gap)
            self.beta = self.beta + self.factor*self.gap
        elif self.type == "gap_stoc":

                
            self.update_gap(gap)
            self.beta_max = self.beta_max + self.factor*self.gap
            self.beta = torch.empty_like(torch.tensor([0.0])).uniform_(1,  self.beta_max).item()    
            
        elif self.type == "loss":
            self.update_loss(lss)
            if len(self.loss) >=2:
                d = math.fabs(self.loss[-1] - self.loss[-2])
                if d<= self.threshold:
                    self.beta = self.beta + self.factor*(1/d)
                self.loss = self.loss[-2:]

        elif self.type == "AugmentBetaOnPlateau" and plat == True:
            self.current_epoch = epoch
            current = float(lss)
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1            
            if self.num_bad_epochs > self.patience and self.beta_list[-1] < self.max_beta:
                self.beta = self.beta*self.factor
                self.num_bad_epochs = 0
                self.beta_list.append(self.beta)
                self.list_epoch.append(epoch)
        elif self.type == "triangle":
            if self.beta <= self.max_triangle:
                self.beta = self.beta + self.factor/self.iteration
            else:
                self.beta = self.beta - self.factor/self.iteration
        else:
            self.beta = self.beta


class RandomAnnealings(nn.Module):

    def __init__(self,beta = 1, left_beta = 1, right_beta = 1000, gap = False, factor = 0.05,type = "random" ):
        super().__init__()

        self.left_beta = left_beta
        self.right_beta = right_beta
        self.beta = beta
        self.beta_max = self.beta
        self.type = type
        self.gap = gap
        self.factor = factor
        self.Trigger = False


    def step(self,gap = None):


        if gap is not None and self.gap is True:
            if self.beta >= 100:
                self.Trigger = True
                self.beta_fix = self.beta
            if self.Trigger is True:
                self.beta =  torch.empty_like(torch.tensor([0.0])).uniform_(self.beta_fix/10, self.beta_fix).item()
            else:
                self.beta = self.beta + self.factor*gap
        else:
            self.beta  = torch.empty_like(torch.tensor([0.0])).uniform_(self.left_beta, self.right_beta).item()

            
class Annealing_triangle(nn.Module):

    def __init__(self, beta, factor = 0.5):
        super().__init__()


        self.increase  = True 
        self.factor = factor
        self.dec_factor = self.factor*0.9
        self.beta = beta
        self.beta_max = self.beta
        self.type = "triangle"


    def step(self, gap):
        if  self.increase == True:
            self.beta_max = self.beta_max + self.factor*gap
            self.beta = torch.empty_like(torch.tensor([0.0])).uniform_(1,  self.beta_max).item()  
        else:
            self.beta_max = self.beta_max - self.dec_factor*gap
            self.beta = torch.empty_like(torch.tensor([0.0])).uniform_(1,  self.beta_max).item()                      







            









class AugmentBetaOnPlateau(nn.Module):
    def __init__(self, 
                 beta,
                mode = "min",
                threshold_mode = "abs",
                factor = 10,
                patience = 10,
                threshold = 1e-3,
                max_beta = 1e5):
        super().__init__()
        if factor < 1.0:
            raise ValueError("Factor should be > 1.0")
        


        self.type = "AugmentBetaOnPlateau"
        self.factor = factor 
        self.threshold_mode = threshold_mode
        self.mode = mode
        self.patience = patience 
        self.threshold = threshold 
        self.max_beta = max_beta
        self.beta = beta 
        self.num_bad_epochs = None
        self.best = 1e2 # very high number 
        self.list_epoch = [] 
        self.current_epoch = 0
        self.counter = 0
        self.beta_list = [self.beta]
        self._reset() 

    def _reset(self):
        self.counter = 0
        self.num_bad_epochs = 0
    

    def step(self, metrics, epoch):
        self.current_epoch = epoch
        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience and self.beta_list[-1] < self.max_beta:
            self.augment_beta()
            self.num_bad_epochs = 0
            self.beta_list.append(self.beta)
            self.list_epoch.append(epoch)
        return self.beta

    def is_better(self,a, best):
        if self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1. - self.threshold
            return a < best*rel_epsilon
        if self.mode == "max" and self.threshold_mode == "abs":
            return a > best - self.threshold
        else:
            rel_epsilon = 1. - self.threshold
            return a > best*rel_epsilon
    
    def augment_beta(self):
        self.beta = self.beta*self.factor  
        

    
class AugmentBetaOnPlateau(nn.Module):
    def __init__(self, 
                 beta,
                mode = "min",
                threshold_mode = "abs",
                factor = 10,
                patience = 10,
                threshold = 1e-3,
                max_beta = 1e5):
        super().__init__()
        if factor < 1.0:
            raise ValueError("Factor should be > 1.0")
        


        self.type = "AugmentBetaOnPlateau"
        self.factor = factor 
        self.threshold_mode = threshold_mode
        self.mode = mode
        self.patience = patience 
        self.threshold = threshold 
        self.max_beta = max_beta
        self.beta = beta 
        self.num_bad_epochs = None
        self.best = 1e2 # very high number 
        self.list_epoch = [] 
        self.current_epoch = 0
        self.counter = 0
        self.beta_list = [self.beta]
        self._reset() 

    def _reset(self):
        self.counter = 0
        self.num_bad_epochs = 0
    

    def step(self, metrics, epoch):
        self.current_epoch = epoch
        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience and self.beta_list[-1] < self.max_beta:
            self.augment_beta()
            self.num_bad_epochs = 0
            self.beta_list.append(self.beta)
            self.list_epoch.append(epoch)
        return self.beta

    def is_better(self,a, best):
        if self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1. - self.threshold
            return a < best*rel_epsilon
        if self.mode == "max" and self.threshold_mode == "abs":
            return a > best - self.threshold
        else:
            rel_epsilon = 1. - self.threshold
            return a > best*rel_epsilon
    
    def augment_beta(self):
        self.beta = self.beta*self.factor  
        
def configure_annealings(factorized_configuration, gaussian_configuration):
    if factorized_configuration is None:
        annealing_strategy_bottleneck = None 
    elif "random" in factorized_configuration["annealing"]:
              
       annealing_strategy_bottleneck = RandomAnnealings(beta = factorized_configuration["beta"],  type = factorized_configuration["annealing"], gap = False)
    elif "triangle" in factorized_configuration["annealing"]:
        annealing_strategy_bottleneck = Annealing_triangle(beta = factorized_configuration["beta"], factor = factorized_configuration["gap_factor"])
    elif "none" in factorized_configuration["annealing"]:
         annealing_strategy_bottleneck = None
    else:
        annealing_strategy_bottleneck = Annealings(beta = factorized_configuration["beta"], 
                                    factor = factorized_configuration["gap_factor"], 
                                    type = factorized_configuration["annealing"]
) 
    if gaussian_configuration is None:
        annealing_strategy_gaussian = None 
    elif "random" in gaussian_configuration["annealing"]:
        annealing_strategy_gaussian = RandomAnnealings(beta = gaussian_configuration["beta"],  type = gaussian_configuration["annealing"], gap = False)
    elif "none" in gaussian_configuration["annealing"]:
        annealing_strategy_gaussian = None
    
    elif "triangle" in gaussian_configuration["annealing"]:
        annealing_strategy_gaussian = Annealing_triangle(beta = factorized_configuration["beta"], factor = factorized_configuration["gap_factor"])
    
    else:
        annealing_strategy_gaussian = Annealings(beta = gaussian_configuration["beta"], 
                                    factor = gaussian_configuration["gap_factor"], 
                                    type = gaussian_configuration["annealing"]) 
    

    #print("********************* CONFIGURED ANNEALINGS *****************************")
    #print("*****************************************************************")
    #print(annealing_strategy_bottleneck)
    #print("-----")
    #print(annealing_strategy_gaussian)
    return annealing_strategy_bottleneck, annealing_strategy_gaussian