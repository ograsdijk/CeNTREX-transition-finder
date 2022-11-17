import numpy as np
import numpy.typing as npt


def get_peaks_resolved(
    frequencies: npt.NDArray[np.float64], resolving_frequency: float = 10
) -> list[tuple[list[int], float]]:
    """
    Take a set of transition frequencies and bunch the ones that are within
    `resolving_energy` together

    Args:
        frequencies_sorted (npt.NDArray[np.float64]): transition frequencies
        resolving_energy (float, optional): bunch if transitions within this value.
                                            Defaults to 10.

    Returns:
        list[tuple[list[int], float]]:
            [(list(bunched indices), mean energy of bunched transitions)]
    """
    frequencies_resolved = []
    de_min = resolving_frequency
    ide = 0
    while ide < len(frequencies) - 1:
        de = frequencies[ide + 1] - frequencies[ide]
        de
        if de > de_min:
            frequencies_resolved.append(([ide], frequencies[ide]))
            ide += 1
        else:
            # start bunching peaks together if within resolving_energy
            indices_peaks_unresolved = [ide, ide + 1]
            peaks_unresolved = [frequencies[ide], frequencies[ide + 1]]
            ide += 1
            while True:
                if ide >= len(frequencies) - 1:
                    frequencies_resolved.append(
                        (indices_peaks_unresolved, np.mean(peaks_unresolved))
                    )
                    ide += 1
                    break
                de = frequencies[ide + 1] - frequencies[ide]
                if de > de_min:
                    # if no more transitions within de, take the mean of the bunched
                    # frequencies and add the indices and mean to the resolved lists
                    frequencies_resolved.append(
                        (indices_peaks_unresolved, np.mean(peaks_unresolved))
                    )
                    ide += 1
                    break
                else:
                    indices_peaks_unresolved.append(ide)
                    peaks_unresolved.append(frequencies[ide])
                    ide += 1

    if ide == len(frequencies) - 1:
        frequencies_resolved.append(([len(frequencies) - 1], frequencies[-1]))

    return frequencies_resolved


def find_overlap_searchsorted(
    frequencies_measured: npt.NDArray[np.float64],
    frequencies_hamiltonian: npt.NDArray[np.float64],
    resolving_frequency: float,
) -> tuple[list[list[int]], list[float]]:
    """
    Find the closest transtion energy matches between a set of measured transitions and
    transitions calculated from the hamiltonian.

    Args:
        frequencies_measured (npt.NDArray[np.float64]): measured transitions
        frequencies_hamiltonian (npt.NDArray[np.float64]): calculated transitions
        resolving_frequency (float): frequency width within which to bunch transitions

    Returns:
        tuple[list[list[int]], list[float]]:
            list[list[bunched indices of hamiltonian matching measured transitions]],
            list[mean energies of matched transitions]
    """
    peaks_measured_resolved = get_peaks_resolved(
        frequencies_measured, resolving_frequency
    )
    frequencies_measured_resolved = np.array([p[1] for p in peaks_measured_resolved])
    frequencies_measured_resolved -= frequencies_measured_resolved[0]

    peaks_hamiltonian_resolved = get_peaks_resolved(
        frequencies_hamiltonian, resolving_frequency
    )
    frequencies_hamiltonian_resolved = np.array(
        [p[1] for p in peaks_hamiltonian_resolved]
    )

    nr_peaks_resolved = len(frequencies_measured_resolved)

    residuals = []
    indices_tracked = []
    for ids in range(len(frequencies_hamiltonian_resolved) - nr_peaks_resolved):
        hamiltonian_spectrum = (
            frequencies_hamiltonian_resolved[ids:]
            - frequencies_hamiltonian_resolved[ids]
        )

        index_sorted = np.argsort(hamiltonian_spectrum)
        hamiltonian_spectrum_sorted = hamiltonian_spectrum[index_sorted]

        idx1 = np.searchsorted(
            hamiltonian_spectrum_sorted, frequencies_measured_resolved
        )
        idx2 = np.clip(idx1 - 1, 0, len(hamiltonian_spectrum_sorted) - 1)

        diff1 = hamiltonian_spectrum_sorted[idx1] - frequencies_measured_resolved
        diff2 = frequencies_measured_resolved - hamiltonian_spectrum_sorted[idx2]

        indices = index_sorted[np.where(diff1 <= diff2, idx1, idx2)]
        res = frequencies_measured_resolved - hamiltonian_spectrum[indices]
        residuals.append(np.sqrt(np.sum(res**2)))
        indices_tracked.append(indices + ids)

    idm = np.argmin(residuals)

    indices = [peaks_hamiltonian_resolved[idx][0] for idx in indices_tracked[idm]]
    energies = [peaks_hamiltonian_resolved[idx][1] for idx in indices_tracked[idm]]
    return indices, energies, [p[1] for p in peaks_measured_resolved]
