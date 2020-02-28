import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environments.environment import Environment


class AddEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim):
        super(AddEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, 100)
        self.l2 = nn.Linear(100, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x


class AddEnv(Environment):
    """Class that represents a list environment. It represents a list of size length of digits. The digits are 10-hot-encoded.
    There are two pointers, each one pointing on a list element. Both pointers can point on the same element.

    The environment state is composed of a scratchpad of size length x 10 which contains the list elements encodings
    and of the two pointers positions.

    An observation is composed of the two encoding of the elements at both pointers positions.

    Primary actions may be called to move pointers and swap elements at their positions.

    We call episode the sequence of actions applied to the environment and their corresponding states.
    The episode stops when the list is sorted.
    """

    def __init__(self, length=10, encoding_dim=32):

        assert length > 0, "length must be a positive integer"
        self.length = length
        self.scratchpad_ints_input_1 = np.zeros((length,), dtype=int) #[0]=0, others: randomly between 0-9
        self.scratchpad_ints_input_2 = np.zeros((length,), dtype=int) #[0]=0, others: randomly between 0-9
        self.scratchpad_ints_carry = np.zeros((length,), dtype=int)   #[length-1]=0, others = -1?
        self.scratchpad_ints_output = np.zeros((length,), dtype=int)  #[:] = -1?
        self.p1_pos = 0
        self.p2_pos = 0
        self.p_o_pos = 0
        self.p_c_pos = 0
        self.encoding_dim = encoding_dim
        self.has_been_reset = False

        self.programs_library = {#level - 0 atomic operations
                                'PTR_1_LEFT': {'level': 0, 'recursive': False},
                                'STOP': {'level': -1, 'recursive': False},
                                'PTR_2_LEFT': {'level': 0, 'recursive': False},
                                'PTR_O_LEFT': {'level': 0, 'recursive': False},
                                'PTR_C_LEFT': {'level': 0, 'recursive': False},
                                'PTR_C_RIGHT': {'level': 0, 'recursive': False},
                                'WRITE_OUTPUT': {'level': 0, 'recursive': False},
                                'WRITE_CARRY': {'level': 0, 'recursive': False},
                                #level - 1 operations
                                'CARRY': {'level': 1, 'recursive': False},
                                #level - 2 operation
                                #'RESET': {'level': 2, 'recursive': False},
                                'ADD_1': {'level': 2, 'recursive': False},
                                'LSHIFT': {'level': 2, 'recursive': False},
                                #level - 3 operation
                                'ADD': {'level': 3, 'recursive': False}}
        for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = idx

        self.prog_to_func = {'STOP': self._stop,
                            'PTR_1_LEFT': self._ptr_1_left,
                            'PTR_2_LEFT': self._ptr_2_left,
                            'PTR_O_LEFT': self._ptr_o_left,
                            'PTR_C_LEFT': self._ptr_c_left,
                            'PTR_C_RIGHT': self._ptr_c_right,
                            'WRITE_CARRY': self._write_carry,
                            'WRITE_OUTPUT': self._write_output
                            }

        self.prog_to_precondition = {'ADD': self._add_precondition,
                                    'ADD_1': self._add_1_precondition,
                                    'LSHIFT': self._lshift_precondition,
                                    'CARRY': self._carry_precondition,
                                    #'RESET': self._reset_precondition,
                                    'STOP': self._stop_precondition,
                                    'PTR_1_LEFT': self._ptr_1_left_precondition,
                                    'PTR_2_LEFT': self._ptr_2_left_precondition,
                                    'PTR_O_LEFT': self._ptr_o_left_precondition,
                                    'PTR_C_LEFT': self._ptr_c_left_precondition,
                                    'PTR_C_RIGHT': self._ptr_c_right_precondition,
                                    'WRITE_OUTPUT': self._write_ouptut_precondition,
                                    'WRITE_CARRY': self._write_carry_precondition
                                    }

        self.prog_to_postcondition = {
                                    'CARRY': self._carry_postcondition,
                                    'LSHIFT': self._lshift_postcondition,
                                    #'RESET': self._reset_postcondition,
                                    'ADD_1': self._add_1_postcondition,
                                    'ADD': self._add_postcondition}

        super(AddEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)

    def _ptr_1_left(self):
        """Move pointer 1 to the left."""
        if self.p1_pos > 0:
            self.p1_pos -= 1

    def _ptr_1_left_precondition(self):
        return self.p1_pos > 0

    def _stop(self):
        """Do nothing. The stop action does not modify the environment."""
        pass

    def _stop_precondition(self):
        return True

    def _ptr_2_left(self):
        """Move pointer 2 to the left."""
        if self.p2_pos > 0:
            self.p2_pos -= 1

    def _ptr_2_left_precondition(self):
        return self.p2_pos > 0

    def _ptr_o_left(self):
        """Move pointer ouptut to the left."""
        if self.p_o_pos > 0:
            self.p_o_pos -= 1

    def _ptr_o_left_precondition(self):
        return self.p_o_pos > 0

    def _ptr_c_left(self):
        """Move pointer carry to the left."""
        if self.p_c_pos > 0:
            self.p_c_pos -= 1

    def _ptr_c_left_precondition(self):
        return self.p_c_pos > 0

    def _ptr_c_right(self):
        """Move pointer carry to the right."""
        if self.p_c_pos < (self.length - 1):
            self.p_c_pos += 1

    def _ptr_c_right_precondition(self):
        return self.p_c_pos < self.length-1

    def _write_output(self):
        """Write output value from the current pointer's value"""
        output = self.scratchpad_ints_input_1[self.p1_pos]
        output += self.scratchpad_ints_input_2[self.p2_pos]
        output += self.scratchpad_ints_carry[self.p_c_pos]
        if output >= 10:
            self.scratchpad_ints_output[self.p_o_pos] = output - 10
        else:
            self.scratchpad_ints_output[self.p_o_pos] = output

    def _write_ouptut_precondition(self):
        """All the pointers should be at the same position"""
        bool = self.p1_pos == self.p2_pos
        bool &= self.p2_pos == self.p_c_pos
        bool &= self.p_c_pos == self.p_o_pos
        return bool

    def _write_carry(self):
        """Write carry value from the current pointers' values"""
        output = self.scratchpad_ints_input_1[self.p1_pos]
        output += self.scratchpad_ints_input_2[self.p2_pos]
        output += self.scratchpad_ints_carry[self.p_c_pos + 1]
        if output >= 10:
            self.scratchpad_ints_carry[self.p_c_pos] = 1
        else:
            self.scratchpad_ints_carry[self.p_c_pos] = 0

    def _write_carry_precondition(self):
        """The carry pointer should be one step ahead of all the other pointers """
        bool = self.p1_pos == self.p2_pos
        bool &= self.p2_pos == self.p_o_pos
        bool &= (self.p_o_pos - 1) == self.p_c_pos
        return bool

    def _lshift_precondition(self):
        # TODO: use "and" to have tighter precondition?
        #return self.p1_pos > 0 or self.p2_pos > 0 or self.p_o_pos > 0 or self.p_c_pos > 0
        #return self.p1_pos > 0 and self.p2_pos > 0 and self.p_o_pos > 0 and self.p_c_pos > 0
        return self.p1_pos==self.p2_pos and self.p2_pos==self.p_o_pos and \
               self.p_o_pos==self.p_c_pos and self.p_o_pos > 0

    def _lshift_postcondition(self, init_state, state):
        init_scratchpad_ints_input_1, init_scratchpad_ints_input_2, \
        init_scratchpad_ints_carry, init_scratchpad_ints_output, \
        init_p1_pos, init_p2_pos, init_p_c_pos, init_p_o_pos = init_state

        scratchpad_ints_input_1, scratchpad_ints_input_2,\
        scratchpad_ints_carry, scratchpad_ints_output, \
        p1_pos, p2_pos, p_c_pos, p_o_pos = state

        bool = np.array_equal(init_scratchpad_ints_input_1, scratchpad_ints_input_1)
        bool &= np.array_equal(init_scratchpad_ints_input_2, scratchpad_ints_input_2)
        bool &= np.array_equal(init_scratchpad_ints_carry, scratchpad_ints_carry)
        bool &= np.array_equal(init_scratchpad_ints_output, scratchpad_ints_output)
        if init_p1_pos > 0:
            bool &= p1_pos == (init_p1_pos-1)
        else:
            bool &= p1_pos == init_p1_pos
        if init_p2_pos > 0:
            bool &= p2_pos == (init_p2_pos-1)
        else:
            bool &= p2_pos == init_p2_pos
        if init_p_c_pos > 0:
            bool &= p_c_pos == (init_p_c_pos-1)
        else:
            bool &= p_c_pos == init_p_c_pos
        if init_p_o_pos > 0:
            bool &= p_o_pos == (init_p_o_pos-1)
        else:
            bool &= p_o_pos == init_p_o_pos
        return bool

    def _carry_precondition(self):
        """All the pointers should be at the same position
           All the pointers should be at > 0 position"""
        bool = self.p1_pos == self.p2_pos
        bool &= self.p2_pos == self.p_c_pos
        bool &= self.p_c_pos == self.p_o_pos
        bool &= (self.p_o_pos > 0)
        return bool

    def _carry_postcondition(self, init_state, state):
        init_scratchpad_ints_input_1, init_scratchpad_ints_input_2, \
        init_scratchpad_ints_carry, init_scratchpad_ints_output, \
        init_p1_pos, init_p2_pos, init_p_c_pos, init_p_o_pos = init_state

        # scratchpad_ints_input_1, scratchpad_ints_input_2, \
        # scratchpad_ints_carry, scratchpad_ints_output, \
        # p1_pos, p2_pos, p_c_pos, p_o_pos = state

        # if init_p1_pos!=p1_pos or init_p2_pos!=p2_pos or init_p_c_pos!=p_c_pos or init_p_o_pos!=p_o_pos:
        #     return False

        # if np.array_equal(init_scratchpad_ints_input_1, scratchpad_ints_input_1):
        #     return False

        # if np.array_equal(init_scratchpad_ints_input_2, scratchpad_ints_input_2):
        #     return False

        # if np.array_equal(init_scratchpad_ints_output, scratchpad_ints_output):
        #     return False

        new_scratchpad_ints_carry = np.copy(init_scratchpad_ints_carry)

        output = init_scratchpad_ints_input_1[init_p1_pos]
        output += init_scratchpad_ints_input_2[init_p2_pos]
        output += init_scratchpad_ints_carry[init_p_c_pos]
        # Precondition had ruled out the condition that init_p_c_pos=0
        if output >= 10:
            new_scratchpad_ints_carry[init_p_c_pos - 1] = 1
        else:
            new_scratchpad_ints_carry[init_p_c_pos - 1] = 0

        # if np.array_equal(new_scratchpad_ints_carry, scratchpad_ints_carry):
        #     return False

        # Check is the operation has already done
        updated = init_scratchpad_ints_carry[init_p_c_pos - 1] != new_scratchpad_ints_carry[init_p_c_pos - 1]

        new_state = (init_scratchpad_ints_input_1, init_scratchpad_ints_input_2,
                    new_scratchpad_ints_carry, init_scratchpad_ints_output,
                    init_p1_pos, init_p2_pos, init_p_c_pos, init_p_o_pos)
        return self.compare_state(state, new_state) and updated
        # return updated

    def _add_1_precondition(self):
        """All the pointers should be at the same position
           All the pointers should be at > 0 position"""
        bool = self.p1_pos == self.p2_pos
        bool &= self.p2_pos == self.p_c_pos
        bool &= self.p_c_pos == self.p_o_pos
        bool &= (self.p_o_pos > 0)
        return bool

    def _add_1_postcondition(self, init_state, state):
        init_scratchpad_ints_input_1, init_scratchpad_ints_input_2, \
        init_scratchpad_ints_carry, init_scratchpad_ints_output, \
        init_p1_pos, init_p2_pos, init_p_c_pos, init_p_o_pos = init_state

        # scratchpad_ints_input_1, scratchpad_ints_input_2, \
        # scratchpad_ints_carry, scratchpad_ints_output, \
        # p1_pos, p2_pos, p_c_pos, p_o_pos = state

        # if init_p1_pos!=p1_pos or init_p2_pos!=p2_pos or init_p_c_pos!=p_c_pos or init_p_o_pos!=p_o_pos:
        #     return False

        # if np.array_equal(init_scratchpad_ints_input_1, scratchpad_ints_input_1):
        #     return False

        # if np.array_equal(init_scratchpad_ints_input_2, scratchpad_ints_input_2):
        #     return False

        new_scratchpad_ints_carry = np.copy(init_scratchpad_ints_carry)
        new_scratchpad_ints_output = np.copy(init_scratchpad_ints_output)

        output = init_scratchpad_ints_input_1[init_p1_pos]
        output += init_scratchpad_ints_input_2[init_p2_pos]
        output += init_scratchpad_ints_carry[init_p_c_pos]
        # Carry
        if output >= 10:
            new_scratchpad_ints_carry[init_p_c_pos - 1] = 1
        else:
            new_scratchpad_ints_carry[init_p_c_pos - 1] = 0
        # Output
        if output >= 10:
            new_scratchpad_ints_output[init_p_o_pos] = output - 10
        else:
            new_scratchpad_ints_output[init_p_o_pos] = output       

        # if np.array_equal(new_scratchpad_ints_carry, scratchpad_ints_carry):
        #     return False

        # if np.array_equal(new_scratchpad_ints_output, scratchpad_ints_output):
        #     return False

        # Check is the operation has already done
        updated = init_scratchpad_ints_carry[init_p_c_pos - 1] != new_scratchpad_ints_carry[init_p_c_pos - 1]
        updated &= init_scratchpad_ints_output[init_p_o_pos] != new_scratchpad_ints_output[init_p_o_pos]

        new_state = (init_scratchpad_ints_input_1, init_scratchpad_ints_input_2,
                    new_scratchpad_ints_carry, new_scratchpad_ints_output,
                    init_p1_pos, init_p2_pos, init_p_c_pos, init_p_o_pos)
        return self.compare_state(state, new_state) and updated
        # return updated

    def _add_precondition(self):
        """All the pointers should be at the initial position"""
        bool = self.p1_pos == self.p2_pos
        bool &= self.p2_pos == self.p_c_pos
        bool &= self.p_c_pos == self.p_o_pos
        bool &= self.p_o_pos == self.length-1
        return bool

    def _add_postcondition(self, init_state, state):
        # Check the output vector has the correct answer
        scratchpad_ints_input_1, scratchpad_ints_input_2, \
        scratchpad_ints_carry, scratchpad_ints_output, \
        p1_pos, p2_pos, p_c_pos, p_o_pos = state

        bool = True
        temp_carry = 0
        correct_output = -1
        for index in reversed(range(1, self.length)):
            output = scratchpad_ints_input_1[index] + scratchpad_ints_input_2[index] + temp_carry
            # Output
            if output >= 10:
                correct_output = output - 10
                temp_carry = 1
            else:
                correct_output = output
                temp_carry = 0
            
            if correct_output != scratchpad_ints_output[index]:
                bool = False
                break

        # check if list is sorted
        return bool

    # RESET
    # def _reset_precondition(self):
    #     bool = True
    #     return bool

    # def _reset_postcondition(self, init_state, state):
    #     init_scratchpad_ints, init_p1_pos, init_p2_pos = init_state
    #     scratchpad_ints, p1_pos, p2_pos = state
    #     bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
    #     bool &= (p1_pos == 0 and p2_pos == 0)
    #     return bool

    def _one_hot_encode(self, digit, basis=10):
        """One hot encode a digit with basis.

        Args:
          digit: a digit (integer between 0 and 9)
          basis:  (Default value = 10)

        Returns:
          a numpy array representing the 10-hot-encoding of the digit

        """
        encoding = np.zeros(basis)
        encoding[digit] = 1
        return encoding

    def _one_hot_decode(self, one_encoding):
        """Returns digit associated to a one hot encoding.

        Args:
          one_encoding: numpy array representing the 10-hot-encoding of a digit.

        Returns:
          the digit encoded in one_encoding

        """
        return np.argmax(one_encoding)

    def reset_env(self):
        """Reset the environment. The two input values are draw randomly.
        The pointers are initialized at position length-1 (at right position of the list).
        The carry/output lists are initialized by value -1
        """
        self.scratchpad_ints_input_1 = np.random.randint(10, size=self.length)
        self.scratchpad_ints_input_1[0] = 0
        self.scratchpad_ints_input_2 = np.random.randint(10, size=self.length)
        self.scratchpad_ints_input_2[0] = 0

        self.scratchpad_ints_carry = np.full((self.length,), -1, dtype=int)
        self.scratchpad_ints_carry[-1] = 0
        self.scratchpad_ints_output = np.full((self.length,), -1, dtype=int)

        current_task_name = self.get_program_from_index(self.current_task_index)
        if current_task_name == 'ADD':
            init_pointers_pos1 = self.length - 1
            init_pointers_pos2 = self.length - 1
            init_pointers_pos_o = self.length - 1
            init_pointers_pos_c = self.length - 1
        # elif current_task_name == 'RESET':
        #     while True:
        #         init_pointers_pos1 = int(np.random.randint(0, self.length))
        #         init_pointers_pos2 = int(np.random.randint(0, self.length))
        #         if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0):
        #             break
        elif current_task_name == 'ADD_1':
            init_pointers_pos1 = int(np.random.randint(1, self.length))
            init_pointers_pos2 = init_pointers_pos1
            init_pointers_pos_o = init_pointers_pos1
            init_pointers_pos_c = init_pointers_pos1
        elif current_task_name == 'LSHIFT':
            # while True:
            #     init_pointers_pos1 = int(np.random.randint(0, self.length))
            #     init_pointers_pos2 = int(np.random.randint(0, self.length))
            #     init_pointers_pos_o = int(np.random.randint(0, self.length))
            #     init_pointers_pos_c = int(np.random.randint(0, self.length))
            #     if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0 \
            #     and init_pointers_pos_o == 0 and init_pointers_pos_c == 0):
            #         break
            init_pointers_pos1 = int(np.random.randint(1, self.length))
            init_pointers_pos2 = init_pointers_pos1
            init_pointers_pos_o = init_pointers_pos1
            init_pointers_pos_c = init_pointers_pos1
        elif current_task_name == 'CARRY':
            # All pointers' positions should be > 0
            init_pointers_pos1 = int(np.random.randint(1, self.length))
            init_pointers_pos2 = init_pointers_pos1
            init_pointers_pos_o = init_pointers_pos1
            init_pointers_pos_c = init_pointers_pos1
        else:
            raise NotImplementedError('Unable to reset env for this program...')

        self.p1_pos = init_pointers_pos1
        self.p2_pos = init_pointers_pos2
        self.p_c_pos = init_pointers_pos_c
        self.p_o_pos = init_pointers_pos_o
        self.has_been_reset = True

        # print("reset_env(): %s, input1: %s, input2: %s, carry: %s, output: %s, p1_pos: %d, p2_pos: %d, p_c_pos: %d, p_o_pos: %d"\
        #      %(current_task_name, self.scratchpad_ints_input_1, self.scratchpad_ints_input_2,\
        #      self.scratchpad_ints_carry, self.scratchpad_ints_output,\
        #      self.p1_pos, self.p2_pos, self.p_c_pos, self.p_o_pos))

    def get_state(self):
        """Returns the current state.

        Returns:
            the environment state

        """
        assert self.has_been_reset, 'Need to reset the environment before getting states'
        return np.copy(self.scratchpad_ints_input_1), np.copy(self.scratchpad_ints_input_2),\
               np.copy(self.scratchpad_ints_carry), np.copy(self.scratchpad_ints_output),\
               self.p1_pos, self.p2_pos, self.p_c_pos, self.p_o_pos

    def get_observation(self):
        """Returns an observation of the current state.

        Returns:
            an observation of the current state
        """
        assert self.has_been_reset, 'Need to reset the environment before getting observations'

        p1_val = self.scratchpad_ints_input_1[self.p1_pos]
        p2_val = self.scratchpad_ints_input_2[self.p2_pos]
        p_c_val = self.scratchpad_ints_carry[self.p_c_pos]
        p_o_val = self.scratchpad_ints_output[self.p_o_pos]
        is_added = int(self._is_added())
        pointers_same_pos = int(self.p1_pos == self.p2_pos and self.p2_pos == self.p_c_pos and self.p_c_pos == self.p_o_pos)
        pt_1_left = int(self.p1_pos == 0)
        pt_2_left = int(self.p2_pos == 0)
        pt_c_left = int(self.p_c_pos == 0)
        pt_o_left = int(self.p_o_pos == 0)
        pt_1_right = int(self.p1_pos == (self.length - 1))
        pt_2_right = int(self.p2_pos == (self.length - 1))
        pt_c_right = int(self.p_c_pos == (self.length - 1))
        pt_o_right = int(self.p_o_pos == (self.length - 1))
        p1p2 = np.eye(10)[[p1_val, p2_val, p_c_val, p_o_val]].reshape(-1)
        bools = np.array([
            pt_1_left,
            pt_1_right,
            pt_2_left,
            pt_2_right,
            pt_c_left,
            pt_c_right,
            pt_o_left,
            pt_o_right,
            pointers_same_pos,
            is_added
        ])
        return np.concatenate((p1p2, bools), axis=0)

    def get_observation_dim(self):
        """

        Returns:
            the size of the observation tensor
        """
        #return 2 * 10 + 6
        return 4 * 10 + 10

    def reset_to_state(self, state):
        """

        Args:
          state: a given state of the environment
        reset the environment is the given state

        """
        self.scratchpad_ints_input_1 = state[0].copy()
        self.scratchpad_ints_input_2 = state[1].copy()
        self.scratchpad_ints_carry = state[2].copy()
        self.scratchpad_ints_output = state[3].copy()
        self.p1_pos = state[4]
        self.p2_pos = state[5]
        self.p_c_pos = state[6]
        self.p_o_pos = state[7]

    def _is_added(self):
        """Assert is the list is sorted or not.

        Args:

        Returns:
            True if the list is sorted, False otherwise

        """
        # arr = self.scratchpad_ints
        # return np.all(arr[:-1] <= arr[1:])

        bool = True
        temp_carry = 0
        correct_output = -1
        for index in reversed(range(1, self.length)):
            output = self.scratchpad_ints_input_1[index] + self.scratchpad_ints_input_2[index] + temp_carry
            # Output
            if output >= 10:
                correct_output = output - 10
                temp_carry = 1
            else:
                correct_output = output
                temp_carry = 0
            
            if correct_output != self.scratchpad_ints_output[index]:
                bool = False
                break

        # check if list is sorted
        return bool

    def get_state_str(self, state):
        """Print a graphical representation of the environment state"""
        input1 = state[0].copy()  # check
        input2 = state[1].copy()  # check
        carry = state[2].copy()  # check
        output = state[3].copy()  # check
        p1_pos = state[4]
        p2_pos = state[5]
        p_c_pos = state[6]
        p_o_pos = state[7]
        str = 'input1 : {}, input2 : {}, carry : {}, output : {}, p1 : {}, p2 : {}, pc : {}, po : {}'.format(input1, input2, carry, output, p1_pos, p2_pos, p_c_pos, p_o_pos)
        return str

    def compare_state(self, state1, state2):
        """
        Compares two states.

        Args:
            state1: a state
            state2: a state

        Returns:
            True if both states are equals, False otherwise.

        """
        bool = True
        bool &= np.array_equal(state1[0], state2[0])
        bool &= np.array_equal(state1[1], state2[1])
        bool &= np.array_equal(state1[2], state2[2])
        bool &= np.array_equal(state1[3], state2[3])
        bool &= (state1[4] == state2[4])
        bool &= (state1[5] == state2[5])
        bool &= (state1[6] == state2[6])
        bool &= (state1[7] == state2[7])
        return bool
