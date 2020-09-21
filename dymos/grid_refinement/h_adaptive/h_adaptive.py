from ...phase.phase import Phase

import numpy as np


def split_segments(old_seg_ends, B):
    """
    Funcion to compute the new segment ends for the refined grid by splitting the necessary segments

    Parameters
    ----------
    old_seg_ends: np.array
        segment ends of grid on which the problem was solved

    B: np.array of ints
        Number of segments to be split into

    Returns
    -------
    new_segment_ends: np.array
        Segment ends of refined grid

    """
    new_segment_ends = []
    for q in range(0, B.size):
        new_ends = list(np.linspace(old_seg_ends[q], old_seg_ends[q + 1], B[q] + 1))
        new_segment_ends.extend(new_ends[:-1])
    new_segment_ends.extend([1])
    new_segment_ends = np.asarray(new_segment_ends)
    return new_segment_ends


class HAdaptive:
    """
    Grid refinement class for the h grid refinement algorithm.
    This method assumes that all segments in a given phase have the same order, and will raise
    an exception if that is not the case.

    The error of a jth segment on which the states are represented by a polynomial is:

    \begin{align}
        \epsilon_j &\propto segl_{j}^{n + 1}
    \end{align}

    Where $segl_j$ is the relative length of the jth segment and relative segment lengths sum to 2.

    \begin{align}
        \sum_{j=0}^{n-1} segl_{j} &= 2
    \end{align}

    Suppose some desired/acceptable error is specified for the segment, then

    \begin{align}
        \frac{\epsilon_j}{\epsilon_{desired}} &= \frac{segl_j}{segl_{j-new}}^{n + 1}
    \end{align}

    The segment length that achieves the desired error is then

    \begin{align}
        segl_{j-new} &= segl_j \left( \frac{\epsilon_j}{\epsilon_{desired}} \right) ^\frac{-1}{n + 1} \label{eq:h_adaptive_new_seg_length}
    \end{align}

    Because relative segment lengths sum to 2, if

    \begin{align}
        \sum_{j=0}^{n-1} segl_{j-new} &\ge 2 \label{eq:h_adaptive_sum_new_seg_length}
    \end{align}

    then theortically there are enough segments to satisfy the error across the phase, if the segments are distributed correctly.
    In this case, the refinement algorith maintains the existing number of segments but changes their relative lengths.
    Significantly changing the relative size of a single segment has demonstrated poor convergence of the grid refinement algorithm.
    Instead, segment growth/contraction as given by \eqref{eq:h_adaptive_new_seg_length} is limited such that

    \begin{align}
        k_{contract} \le \frac{segl_{j-new}}{segl_{j}} \le k_{expand}
    \end{align}

    Where $k_{contract}$ and $k_{expand}$ are user-defined parameters which limit the contraction or
    expansion of any given segment.

    When \eqref{eq:h_adaptive_sum_new_seg_length} is not satisfied, more segments are needed to achieve the desired error.
    In this case, the maximum error is computed along a segment and a running total of its absolute value is maintained.
    When this is plotted against $\tau_{phase}$, a linear interpolation of the plot can be used and divided into segments such
    that the error accumulated in each segment in the new grid is equally distributed.

    """

    def __init__(self, phases):
        """
        Initialize and compute attributes

        Parameters
        ----------
        phases: Phase
            The Phase object representing the solved phase

        """
        self.phases = phases
        self.error = {}

    def refine(self, refine_results, iter_number):
        """
        Compute the order, number of nodes, and segment ends required for the new grid
        and assigns them to the transcription of each phase.

        Parameters
        ----------
        iter_number: int
            An integer value representing the iteration of the grid refinement

        refine_results : dict
            A dictionary where each key is the path to a phase in the problem, and the
            associated value are various properties of that phase needed by the refinement
            algorithm.  refine_results is returned by check_error.  This method modifies it
            in place, adding the new_num_segments, new_order, and new_segment_ends.

        Returns
        -------
        refined : dict
            A dictionary of phase paths : phases which were refined.

        """
        growth_limit = 1.2
        contraction_limit = 0.9
        e_desired = 1.0E-6

        for phase_path, phase_refinement_results in refine_results.items():
            phase = self.phases[phase_path]
            tx = phase.options['transcription']
            gd = tx.grid_data

            need_refine = phase_refinement_results['need_refinement']


            if not phase.refine_options['refine'] or not np.any(need_refine):
                refine_results[phase_path]['new_order'] = gd.transcription_order
                refine_results[phase_path]['new_num_segments'] = gd.num_segments
                refine_results[phase_path]['new_segment_ends'] = gd.segment_ends
                continue

            # Refinement is needed
            gd = phase.options['transcription'].grid_data
            numseg = gd.num_segments

            refine_seg_idxs = np.where(need_refine)
            P = np.zeros(numseg)

            max_rel_error = refine_results[phase_path]['max_rel_error'][refine_seg_idxs]
            tol = phase.refine_options['tolerance']
            order = gd.transcription_order[refine_seg_idxs]

            P[refine_seg_idxs] = np.log(max_rel_error / tol) / np.log(order)
            P = np.ceil(P).astype(int)

            if gd.transcription == 'gauss-lobatto':
                odd_idxs = np.where(P % 2 != 0)
                P[odd_idxs] += 1

            new_order = gd.transcription_order + P
            B = np.ones(numseg, dtype=int)

            raise_order_idxs = np.where(gd.transcription_order + P <= phase.refine_options['max_order'])
            split_seg_idxs = np.where(gd.transcription_order + P > phase.refine_options['max_order'])

            new_order[raise_order_idxs] = gd.transcription_order[raise_order_idxs] + P[raise_order_idxs]
            new_order[split_seg_idxs] = phase.refine_options['min_order']

            B[split_seg_idxs] = np.around((gd.transcription_order[split_seg_idxs] +
                                           P[split_seg_idxs]) / phase.refine_options['min_order']).astype(int)

            new_order = np.repeat(new_order, repeats=B)
            new_num_segments = int(np.sum(B))
            new_segment_ends = split_segments(gd.segment_ends, B)

            refine_results[phase_path]['new_order'] = new_order
            refine_results[phase_path]['new_num_segments'] = new_num_segments
            refine_results[phase_path]['new_segment_ends'] = new_segment_ends

            tx.options['order'] = new_order
            tx.options['num_segments'] = new_num_segments
            tx.options['segment_ends'] = new_segment_ends
            tx.init_grid()
