# Project1 for EN.520.666 Information Extraction

# 2021 Matthew Ost
# 2021 Ruizhe Huang
# 2022 Zili Huang

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import string

EPS = 1e-6

def load_data(fname):
    alphabet_string = string.ascii_lowercase
    char_list = list(alphabet_string)
    print(char_list)
    char_list.append(' ')
    with open(fname, 'r') as fh:
        content = fh.readline()
    content = content.strip('\n')
    data = []
    for c in content:
        assert c in char_list
        data.append(char_list.index(c))
    return np.array(data) 

def get_init_prob_2states():
    # Define initial transition probability and emission probability
    # for 2 states HMM
    
    T_prob=np.array([[0.26,0.26,0.24,0.24],[0.24,0.24,0.26,0.26],[0.26,0.26,0.24,0.24],[0.24,0.24,0.26,0.26]])

    S_1_1=[0.037]*13
    S_1_2=[0.0371]*13

    S_2_1=[0.0371]*13
    S_2_2=[0.037]*13

    S_1=S_1_1+S_1_2+[0.0367]
    S_2=S_2_1+S_2_2+[0.0367]

    S_3= list(S_1)
    S_4= list(S_1)


    E_prob=np.array([S_1,S_2,S_3,S_4])



    return T_prob, E_prob

class HMM:

    def __init__(self, num_states, num_outputs):
        # Args:
        #     num_states (int): number of HMM states
        #     num_outputs (int): number of output symbols            

        self.states = np.arange(num_states)  # just use all zero-based index
        self.outputs = np.arange(num_outputs)
        self.num_states = num_states
        self.num_outputs = num_outputs

        # Probability matrices
        self.transitions = None
        self.emissions = None

        # additional parameters

        self.alpha=None

        self.data_probability=None

        self.Q=None

    def initialize(self, T_prob, E_prob):
        # Initialize HMM with transition probability T_prob and emission probability
        # E_prob

        # Args:
        #     T_prob (numpy.ndarray): [num_states x num_states] numpy array.
        #     T_prob[i, j] is the transition probability from state i to state j.
        #     E_prob (numpy.ndarray): [num_states x num_outputs] numpy array.
        #     E_prob[i, j] is the emission probability of state i to output jth symbol. 
        self.transitions = T_prob
        self.emissions = E_prob
        self._assert_transition_probs()
        self._assert_emission_probs()

    def _assert_emission_probs(self):
        for s in self.states:
            assert self.emissions[s].sum() - 1 < EPS

    def _assert_transition_probs(self):
        for s in self.states:
            assert self.transitions[s].sum() - 1 < EPS
            assert self.transitions[:, s].sum() - 1 < EPS

    def Baum_Welch(self, max_iter, train_data, test_data):
        # The Baum Welch algorithm to estimate HMM parameters
        # Args:
        #     max_iter (int): maximum number of iterations to train
        #     train_data (numpy.ndarray): train data
        #     test_data (numpy.ndarray): test data
        #
        # Returns:
        #     info_dict (dict): dictionary containing information to visualize

        average_log_likelihood=[]
        all_test=[]

        emission_0_0=[]
        emission_1_0=[]
        emission_2_0=[]
        emission_3_0=[]
        emission_0_13=[]
        emission_1_13=[]
        emission_2_13=[]
        emission_3_13=[]

        info_dict = {"log_likelihood_during_training":average_log_likelihood,
        "log_likelihood_during_testing":all_test,"emission_0_a":emission_0_0,"emission_1_a":emission_1_0,
        "emission_0_n":emission_0_13,"emission_1_n":emission_1_13,"emission_2_a":emission_2_0,"emission_3_a":emission_3_0,"emission_2_n":emission_2_13,"emission_3_n":emission_3_13}




        for it in range(max_iter):

            print(it)
            # Implement the Baum-Welch algorithm here

            # Get the initial alpha and initial beta 

            individual_prob=1/self.num_states
            alpha=[[1.0,0.0,0.0,0.0]]
            beta=[[1.0]*self.num_states]

            # keep track of the trellis probs for parameters estimation
            all_trellis_probs=[]

            # keep track of the data likelihood
            data_probability=0

            # Q

            q_list=[1.0]

            # The forward pass
            for i in range(len(train_data)):
                # Extract the previous alpha
                alpha_out=alpha[i]
                # compute the alpha for every single states 
                current_alpha=[]
                for j in range(self.num_states):
                    # Gather alpha_out from all previous states
                    alpha_j_at_time_i=0
                    for k in range(self.num_states):
                        # keep track of the state that emits alpha and the state that receive alpha
                        emit_state=k
                        receive_state=j
                        # compute the updated alpha 
                        alpha_j_at_time_i+=alpha_out[k]*self.transitions[emit_state,receive_state]*self.emissions[receive_state,train_data[i]]
                    # Put all alpha of each individual states back 
                    current_alpha.append(alpha_j_at_time_i)

                normalized_q=sum(current_alpha)

                # store the q so that we can do use it to normalize b later
                q_list.append(normalized_q)

                #print(current_alpha)

                for a in range(len(current_alpha)):
                    current_alpha[a]=current_alpha[a]/normalized_q


                alpha.append(current_alpha)


                data_probability+=np.log(normalized_q)

            self.data_probability=data_probability




            for i in range(len(beta[0])):
                beta[0][i]=beta[0][i]/q_list[-1]




            # The backward pass

            for i in range(len(train_data)):
                # compute the current time (since it is opposite to wheat we have done for alpha)
                current_time=len(train_data)-1-i
                # Extract the previous beta 
                beta_out=beta[i]
                # Compute the beta for every single states
                current_beta=[]
                for j in range(self.num_states):
                    # Gather beta_out from all later states
                    beta_j_at_time_i=0
                    for k in range(self.num_states):
                        # keep track of the state that emits beta and the state that receive beta
                        emit_state=k
                        receive_state=j
                        # Compute the new beta_out
                        beta_j_at_time_i+=beta_out[k]*self.transitions[receive_state,emit_state]*self.emissions[emit_state,train_data[current_time]]
                    current_beta.append(beta_j_at_time_i)

                # normalize the beta
                normalized_q=q_list[-2-i]
                for b in range(len(current_beta)):
                    current_beta[b]=current_beta[b]/normalized_q
                beta.append(current_beta)
                # since we know the all alphas now and the beta at the momemnt, we can evaluates the P(t)
                # generate the palce where we can store the trellis probs.
                trellis_probs=[]
                for state in range(self.num_states):
                    trellis_probs.append([])

                # compute the trellis probs
                for j in range(self.num_states):
                    # k represents the end of the trellis
                    for k in range(self.num_states):
                        current_alpha=alpha[-1-i-1]
                        alpha_at_state_j=current_alpha[j]
                        beta_at_state_k=beta_out[k]
                        p_trellis_start_at_j=alpha_at_state_j*self.transitions[j,k]*self.emissions[k,train_data[-1-i]]*beta_at_state_k
                        trellis_probs[j].append(p_trellis_start_at_j)


                # Transition probability evaluation
                # we have computed the transition probability while doing the backward computation
                all_trellis_probs.append(trellis_probs)

            #print(all_trellis_probs[-1])

            #print(beta[2])
            # Parameter Estimation


            # initialize the trellis_count for each y(output)
            t_count_at_y=[]

            for i in range(self.num_outputs):
                t_count=[]
                for j in range(self.num_states):
                    t_count.append([])
                    for k in range(self.num_states):
                        t_count[j].append(0)
                t_count_at_y.append(t_count)

            ## we need to reverse the trellis back since we computed them in a backward direction


            all_trellis_probs=all_trellis_probs[::-1]


            # iterate through all trellis to fill in the trellis_count
            for time in range(len(all_trellis_probs)):
                trellis=all_trellis_probs[time]
                for i in range(self.num_states):
                    trellis_at_i=trellis[i]
                    # i is the start of the trellis 
                    for j in range(self.num_states):
                        # j is the end of the trellis
                        # cumulate the sum of the trellis to each y 
                        current_y_produced=train_data[time]
                        t_count_at_y[current_y_produced][i][j]+=trellis_at_i[j]



            unnormalized_emission_counts=np.zeros([self.num_states,self.num_outputs])

            for i in range(self.num_states):
                for j in range(self.num_outputs):
                    numerator_for_y_at_i=0.0
                    for k in range(self.num_states):
                        numerator_for_y_at_i+=t_count_at_y[j][k][i]
                    unnormalized_emission_counts[i,j]=numerator_for_y_at_i




            # compute c(t) by summing over all possible y
            for i in range(self.num_states):
                sum_at_i=sum(unnormalized_emission_counts[i])
                for j in range(self.num_outputs):
                    unnormalized_emission_counts[i,j]=unnormalized_emission_counts[i,j]/sum_at_i

            self.emissions=unnormalized_emission_counts


            # compute the transition probability next 

            # remember to modify when changing number of states
            count_t=np.array([[0.0]*self.num_states,[0.0]*self.num_states,[0.0]*self.num_states,[0.0]*self.num_states])

            for i in range(self.num_outputs):
                count_t+=t_count_at_y[i]

            for i in range(self.num_states):
                sum_c_t=sum(count_t[i])
                for j in range(self.num_states):
                    count_t[i,j]=count_t[i,j]/sum_c_t

            self.transitions=count_t


            self.alpha=alpha


            average_log_likelihood.append(data_probability/len(train_data))

            #print("train:",data_probability/len(train_data))




            test_likelihood=0
            alpha_test=[[1.0,0.0,0.0,0.0]]
            for i in range(len(test_data)):
                # Extract the previous alpha
                alpha_out=alpha_test[i]
                # compute the alpha for every single states 
                current_alpha=[]
                for j in range(self.num_states):
                    # Gather alpha_out from all previous states
                    alpha_j_at_time_i=0
                    for k in range(self.num_states):
                        # keep track of the state that emits alpha and the state that receive alpha
                        emit_state=k
                        receive_state=j
                        # compute the updated alpha 
                        alpha_j_at_time_i+=alpha_out[k]*self.transitions[emit_state,receive_state]*self.emissions[receive_state,test_data[i]]
                    # Put all alpha of each individual states back 
                    current_alpha.append(alpha_j_at_time_i)

                normalized_q=sum(current_alpha)

                #print(current_alpha)

                for a in range(len(current_alpha)):
                    current_alpha[a]=current_alpha[a]/normalized_q


                alpha_test.append(current_alpha)


                test_likelihood+=np.log(normalized_q)

            all_test.append(test_likelihood/len(test_data))

            #print("test",test_likelihood/len(test_data))

            emission_0_0.append(self.emissions[0,0])
            emission_1_0.append(self.emissions[1,0])
            emission_0_13.append(self.emissions[0,13])
            emission_1_13.append(self.emissions[1,13])
            emission_2_0.append(self.emissions[2,0])
            emission_3_0.append(self.emissions[3,0])
            emission_2_13.append(self.emissions[2,13])
            emission_3_13.append(self.emissions[3,13])




        #print(info_dict["log_likelihood_during_testing"][-1])
        #print(info_dict["log_likelihood_during_training"][-1])
        
        return info_dict

    def log_likelihood(self, data):
        # Compute the log likelihood of sequence data
        # Args:
        #     data (numpy.ndarray): 
        #
        # Returns:
        #     prob (float): log likelihood of data


        test_likelihood=0
        alpha_test=[[1.0,0.0]]
        for i in range(len(data)):
            # Extract the previous alpha
            alpha_out=alpha_test[i]
            # compute the alpha for every single states 
            current_alpha=[]
            for j in range(self.num_states):
                # Gather alpha_out from all previous states
                alpha_j_at_time_i=0
                for k in range(self.num_states):
                    # keep track of the state that emits alpha and the state that receive alpha
                    emit_state=k
                    receive_state=j
                    # compute the updated alpha 
                    alpha_j_at_time_i+=alpha_out[k]*self.transitions[emit_state,receive_state]*self.emissions[receive_state,data[i]]
                    # Put all alpha of each individual states back 
                current_alpha.append(alpha_j_at_time_i)

            normalized_q=sum(current_alpha)


            for a in range(len(current_alpha)):
                current_alpha[a]=current_alpha[a]/normalized_q

            alpha_test.append(current_alpha)


            test_likelihood+=np.log(normalized_q)
        
        
        return test_likelihood

    def visualize(self, info_dict):

        # print_out

        print(self.emissions)
        print(self.transitions)

        print(info_dict["log_likelihood_during_training"][-1])
        print(info_dict["log_likelihood_during_testing"][-1])




        # Visualize
        plt.plot(info_dict["log_likelihood_during_training"])
        plt.plot(info_dict["log_likelihood_during_testing"])
        plt.show()

        plt.plot(info_dict["emission_0_a"],label="0")
        plt.plot(info_dict["emission_1_a"],label="1")

        plt.show()

        plt.plot(info_dict["emission_0_n"],label="0")
        plt.plot(info_dict["emission_1_n"],label="1")

        plt.show()


        
        


def main():
    n_states = 4
    n_outputs = 27
    train_file, test_file = "textA.txt", "textB.txt"
    max_iter = 600

    ## define initial transition probability and emission probability
    T_prob, E_prob = get_init_prob_2states() 
    #
    ## initial the HMM class
    H = HMM(num_states=n_states, num_outputs=n_outputs)

    ## initialize HMM with the transition probability and emission probability
    H.initialize(T_prob, E_prob)

    # load text file
    train_data, test_data = load_data(train_file), load_data(test_file)

    ## train the parameters of HMM
    info_dict = H.Baum_Welch(max_iter, train_data, test_data)

    ## visualize
    H.visualize(info_dict)

if __name__ == "__main__":
    main()