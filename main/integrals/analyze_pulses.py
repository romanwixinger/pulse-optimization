"""Calculate statistics of the optimized pulses.

Note:
    Before executing this script, one has to run 'main/integrals/minimize_integrals.py' with a valid 'run' that
    corresponds to a file 'configuration/integrals/{run}.json'.
"""

from quantum_gates.utilities import load_config

from pulse_opt.integrals.utilities import load_table_from_pickle, save_table_as_pickle
from pulse_opt.integrals.metrics import metric_lookup
from pulse_opt.integrals.result_enricher import ResultEnricher


def main(run: str):
    """ Executes the statistics calculation on the results table.
    """
    # Load data
    config = load_config(f"integrals/{run}.json")
    df = load_table_from_pickle(run=run)

    # Enrich by computing metrics
    enricher = ResultEnricher(metric_lookup=metric_lookup)
    df = enricher.enrich(df)
    df = enricher.enrich_default(df)
    enricher.report(df)

    # Save
    pass


if __name__ == "__main__":

    runs = [
        'power_test',
    ]

    for run in runs:
        main(run)
