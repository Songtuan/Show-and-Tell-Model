import torch
import torch.nn as nn


class BeamSearch:
    def __init__(self, beam_size, state_machine, end_token_idx, seq_length):
        self.beam_size = beam_size
        self.state_machine = state_machine
        self.end_token_idx = end_token_idx
        self.seq_length = seq_length

    def _step(self, beam_sizes, logprobsf, beam_seq, beam_seq_logprobs, beam_logprobs_sum, time_step, hidden_states):
        '''
        single beam search step
        :param beam_sizes: a list which contain the number of elements in each beam/state
        :param logprobsf: log-probs of current timestep, shape:(beam_size, vocab_size)
        :param beam_seq: prev beam search result, shape: (seq_length, beam_size * beam_num)
        :param beam_seq_logprobs: log-probs of each previous time-step of each beam element,
                                  shape: (seq_length, beam_size * beam_num)
        :param beam_logprobs_sum: the log-prob of current sequence, shape: (beam_size * beam_num)
        :param time_step: current time step
        :param hidden_states: a dict contains previous hidden states, key: str, e.g. 'h1', 'c1', h2', 'c2',
                              value: tensor with shape (beam_size * beam_num, hidden_size)
        :return:
        '''
        # the number of beams/states
        beam_num = len(beam_sizes)

        # candidates is used to store potential elements for each beam/state
        candidates = {i: [] for i in range(beam_num)}

        # cols equal to vocab size
        # which is used to index each word
        cols = logprobsf.shape[-1]

        # rows equal to the total number of elements of all beams
        # which is used to index each beam element
        rows = sum(beam_sizes)

        if time_step == 0:
            assert rows == 1

        for s_idx in range(beam_num):
            # map the state id to state name
            s_t = self.state_machine.state_idx_mapping[s_idx]
            for q in range(rows):
                # iterate through each beam to find the element which
                # can trigger the state transition q -> s_t
                # for each beam, we only take the top beam_size words it generate
                # as out candidates

                s_current = self.state_machine.state_idx_mapping[int(q / self.beam_size)]  # current state
                # fetch the ids of words which can trigger state transition
                transition = self.state_machine.get_transition(s_current, s_t, 'input')
                if transition is None:
                    continue
                trigger_words_idx = transition.word_ids

                # the log-probs distribution of words produced by current beam element/state element
                log_probs = logprobsf[q: q + 1, :]

                # mask the words which cannot trigger state transition
                # to ensure them will not be select as candidates
                mask = torch.ones(1, cols)
                mask[:, trigger_words_idx] = 0
                mask = mask.bool()
                log_probs = log_probs.masked_fill(mask, -1000)

                # if number of trigger words smaller than beam size
                # we select this smaller number of candidates
                k = min(self.beam_size, len(trigger_words_idx))
                probs, idxs = log_probs.topk(k)
                probs = probs.view(-1)
                idxs = idxs.view(-1)

                for i in range(k):
                    #  add trigger words to our candidates
                    local_logprob = probs[i]
                    idx = idxs[i]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob.cpu()
                    candidates[s_idx].append(dict(c=idx, q=q, p=candidate_logprob, r=local_logprob))

        candidates = {s: sorted(candidates[s], key=lambda x: -x['p']) for s in range(beam_num)}
        # reset the number of elements within each beam
        # the number of elements should be the minimum of beam_size and the numbere of candidates
        beam_sizes = [min(self.beam_size, len(candidates[s])) for s in range(beam_num)]

        new_states = {hidden: hidden_states[hidden].clone() for hidden in hidden_states.keys()}
        # copy the previous beam search result
        if time_step >= 1:
            # we''ll need these as reference when we fork beams around
            beam_seq_prev = beam_seq[: time_step].clone()
            beam_seq_logprobs_prev = beam_seq_logprobs[: time_step].clone()

        for idx in range(beam_num * self.beam_size):
            beam_idx = int(idx / self.beam_size)
            # if all elements within beam has been fetched out
            # this will happen when candidate_size < beam_size
            if len(candidates[beam_idx]) == 0:
                continue
            # fetch the top element from beam
            e = candidates[beam_idx].pop(0)
            # match the current element to the previous sequence it generated from
            if time_step >= 1:
                beam_seq[: time_step, idx] = beam_seq_prev[:, e['q']]
                beam_seq_logprobs[: time_step, idx] = beam_seq_logprobs_prev[:, e['q']]

            for hidden in new_states:
                # TODO: need to check further
                new_states[hidden][idx, :] = hidden_states[hidden][e['q'], :]

            beam_seq[time_step, idx] = e['c']
            beam_seq_logprobs[time_step, idx] = e['r']
            beam_logprobs_sum[idx] = e['p']

        hidden_states = new_states

        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_sizes, hidden_states

    def search(self, hidden_states, log_probs, get_logprobs):
        beam_num = len(self.state_machine.get_states())
        beam_seq = torch.LongTensor(self.seq_length, self.beam_size * beam_num).zero_() + (log_probs.size(1) - 1)
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, self.beam_size * beam_num).zero_()
        beam_logprobs_sum = torch.zeros(self.beam_size * beam_num)
        done_beams = []
        beam_sizes = [0] * beam_num
        beam_sizes[0] = 1

        for time_step in range(self.seq_length):
            logprobsf = log_probs.data.float()
            logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

            beam_seq, \
            beam_seq_logprobs, \
            beam_logprobs_sum, \
            beam_sizes, \
            hidden_states = self._step(beam_sizes, logprobsf, beam_seq, beam_seq_logprobs, beam_logprobs_sum, time_step, hidden_states)

            for vix in range(self.beam_size, self.beam_size * beam_num):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[time_step, vix] == self.end_token_idx or time_step == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

                # if the current beam element has not been selected
                # we must ensure the <UNK> token at this position will
                # will not be selected in next round
                elif beam_seq[time_step, vix] == logprobsf.size(1) - 1:
                    beam_logprobs_sum[vix] = -1000

            it = beam_seq[time_step]
            log_probs, _, hidden_states = get_logprobs(it, hidden_states)

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[: self.beam_size]
        return done_beams