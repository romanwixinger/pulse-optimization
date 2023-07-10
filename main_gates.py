from main.gates.simulate_gate import main


if __name__ == "__main__":

    runs = ["standard_configuration"]

    for run in runs:
        print(f"Start run with configuration {run}")
        main(run=run)
