import unittest
from StateMachine import *
from BeamStateMachine import *
from models import *
import torch
import torch.nn as nn


class MyTestCase(unittest.TestCase):
    def test_beam_search(self):
        vocab = {'SOS': 0, 'EOS': 1, 'start': 2, 'do': 3, 'relax': 4, 'cap': 5, 'PAD': 6}
        state_machine = StateMachine(events={'input': InputEvent()})
        state_machine.add_state('init')
        state_machine.add_state('final')
        state_machine.add_transition('init', 'final', 'input', TransitCondition([2]))
        state_machine.add_transition('init', 'init', 'input', TransitCondition([0, 1, 3, 4, 5, 6]))
        state_machine.add_transition('final', 'final', 'input', TransitCondition([0, 1, 2, 3, 4, 5, 6]))
        state_idx_mapping = {0: 'init', 1: 'final'}
        state_machine.add_state_idx_mapping(state_idx_mapping)

        model = ShowTellModel(vocab=vocab, embedd_size=300, attention_size=512, hidden_size=512, state_machine=state_machine)
        model.cuda()
        test_input = torch.rand(2, 3, 224, 224).cuda()
        model.eval()
        seq, _ = model(test_input)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
