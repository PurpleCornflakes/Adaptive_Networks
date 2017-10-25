import numpy as np
import time

np.random.seed(2017)
class AdaptiveNet:

    def __init__(self, N, M, input_red, input_green, desired_red, desired_green):

        self.N = N
        self.M = M
        self.T = 0.4 # threshold
        self.A0 = 2
        self.eta0 = 0.01
        self.input_red = input_red
        self.input_green = input_green
        # desired nodes must be in the last layer
        self.desired_red = desired_red # one dimensional array
        self.desired_green = desired_green
        # current signal. 0 for red, 1 for green
        self.cur_sig = 0
        if self.cur_sig == 0 :
            self.input = input_red
            self.desired = desired_red
        else:
            self.input = input_green
            self.desired = desired_green

        self.net = np.zeros((N,M), dtype = np.int)
        self.positive_series = np.array([])
        # initialize weights
        # every successive 3 in this array corresponds to the 3 lines
        # o o o
        # \ | /
        #   o     
        self.weights = np.random.random((N-1, M*3))
        # normalize
        self.normalize_weight()

        label = ["\033[00;29m^ \033[00m"] * self.M
        for r in self.desired_red:
            label[r] = "\033[01;31m^ \033[00m"
        for g in self.desired_green:
            label[g] = "\033[01;32m^ \033[00m"
        self.label = str.join("", label) + "\n"


    def change_signal(self):
        if self.cur_sig == 0:
            self.input = self.input_green
            self.desired = self.desired_green
            self.cur_sig = 1
        else:
            self.input = self.input_red
            self.desired = self.desired_red
            self.cur_sig = 0

    def __get_new_net(self):
        """
        computes new network. the old one is kept
        """
        new_net = self.net[:]
        tmp = self.__expand(self.net)
        tmp *= self.weights
        bf_threshold = np.asarray([np.sum(tmp[:, j*3:j*3+3], axis = 1) for j in range(self.M)]).T
        # update according to threshold
        new_net[1:,:] = np.where(bf_threshold <= self.T, 0, 1)
        new_net[self.input.T[0], self.input.T[1]] = 1
        return new_net

    def one_iterate(self):
        """
        call get_new_net and update weights
        """
        is_positive = False
        new_net = self.__get_new_net()
        A = np.sum(new_net[-1,:])
        if np.all(new_net[-1, self.desired] == 1):
            r = 0.1
            is_positive = True
        else:
            r = -0.1
        self.T += np.sign(A-self.A0)*0.01/self.N

        n_old = self.__expand(self.net)

        n_new = np.zeros((self.N-1, self.M*3))
        for j in range(self.M):
            n_new[:, j*3:j*3+3] = np.array([new_net[1:, [ j,j,j ]]])


        eta = np.random.uniform(-self.eta0, self.eta0)
        self.weights += ( r * self.weights * ( 1-self.weights) + eta ) * n_old * n_new 
        self.normalize_weight()
        self.net = new_net

        if len(self.positive_series) == 250:
            self.positive_series = np.delete(self.positive_series,[0])
        self.positive_series = np.append(self.positive_series, is_positive)
            
        return is_positive

    def iterate(self, times, pause_time):

        single_signal_timer = 0
        consistent_positive = 0
        self.print_out(0)
        for i in range(times):
            time.sleep(pause_time)
            is_positive = self.one_iterate()
            single_signal_timer += 1
            if is_positive:
                consistent_positive += 1
            else:
                consistent_positive = 0

            if consistent_positive >= 250 \
                or single_signal_timer >= 2000:
                self.change_signal()
                single_signal_timer = 0
                consistent_positive = 0
            self.print_out(i+1)
        print("\033[{}B".format(self.N+1))

    def __color_node(self, indx):
        color = "\033[01;31m" if self.cur_sig == 0 else "\033[01;32m"
        if indx in self.input.tolist():
            return color + 'o ' + "\033[00m"
        elif self.net[indx[0], indx[1]] == 1 :
            return "\033[01;33mo \033[00m"
        else:
            return "\033[01;30m- \033[00m"

    def print_out(self, t):
        
        lines = ""
        for i in range(self.N):
            lines += str.join( "", map(self.__color_node, [[i, j] for j in range(self.M)] )) + "\n"
        lines += self.label
        lines += "iter = {0:d}, threshold = {1:.3f}, p = {2:.3f}".format(t, self.T, np.mean(self.positive_series))+"\n"
        print(lines, end="")
        print("\033[{}A".format(self.N+3))


    def __expand(self, in_array):
        """
        convolveingly expand the in_array, with PBC.
        expand rule: 0,1,2,3,4 --> (4,0,1),(0,1,2),(1,2,3),(2,3,4),(3,4,0)
        """
        tmp = np.zeros((self.N-1, self.M*3), dtype = np.float64)
        for j in range(self.M):
            tmp[:, j*3:j*3+3] = in_array[0:-1, [ (j-1)%self.M, j, (j+1)%self.M ]]
        return np.asarray(tmp, dtype = np.float64)

    def normalize_weight(self):
        for i in range(self.N-1):
            for j in range(self.M):
                jj = 1+j*3
                ww = self.weights[i,[(jj-2)%(self.M*3),jj,(jj+2)%(self.M*3)]]
                self.weights[i,[(jj-2)%(self.M*3),jj,(jj+2)%(self.M*3)]] = ww/np.sum(ww)





if __name__ == "__main__":


    input_red = np.array([[0,5],[1,13],[2,3],[3,7],[3,12],[4,9],[6,3],[6,9],[7,11],[9,2],[9,5],[9,10],[9,15],[10,9],[11,5],[11,13]])
    input_green = np.array([[2,1],[2,11],[4,1],[4,15],[5,5],[5,11],[5,13],[6,7],[8,8],[8,13],[10,1],[10,6],[11,4],[12,1],[12,7],[12,10]])
    desired_red = np.array([3,6])
    desired_green = np.array([10,14])

    a = AdaptiveNet(32, 32, input_red, input_green, desired_red, desired_green)
    a.iterate(times = 100000, pause_time = 0)
