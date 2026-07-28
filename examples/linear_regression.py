import dlf
import matplotlib.pyplot as plt
import numpy as np


class regressionTestModel(dlf.Model):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = dlf.Linear(in_dim, out_dim)
        self.add_layer(self.l1)

    def forward(self, x):
        return self.l1.forward(x)


def main():

    # -----------------------------
    # Training data
    # -----------------------------
    x1 = np.random.uniform(0, 5, 100)
    x2 = np.random.uniform(0, 5, 100)
    x_data = dlf.Tensor([[float(a), float(b)] for a,b in zip(x1,x2)])

    # y = 5*x1 - x2 + 1
    noise = np.random.normal(0, 2.0, 100)
    y_true = 5 * x1 - x2 + 1 + noise
    y_data = dlf.Tensor([[float(y)] for y in y_true])

    model = regressionTestModel(in_dim=2, out_dim=1)
    criterion = dlf.MSEloss()

    model.train()

    epochs = 201
    losses = []

    # -----------------------------
    # Generate random "true" data
    # -----------------------------
    np.random.seed(42)

    # x values for the prediction line
    x_line = np.linspace(0, 5, 100)

    # -----------------------------
    # Interactive plotting
    # -----------------------------
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for epoch in range(epochs):

        # Train one step
        model.step(x_data, y_data)

        pred = model.forward(x_data)
        loss = criterion.forward(y_data, pred)
        
        loss_val = loss.data()[0]
        losses.append(loss_val)

        # Update plots every 5 epochs
        if epoch % 5 == 0:

            axes[0].cla()
            axes[1].cla()

            # ---------------- Loss Plot ----------------
            axes[0].plot(losses, color="tab:blue")
            axes[0].set_xlim(0, epochs)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("MSE Loss")
            axes[0].set_title("Training Loss")
            axes[0].grid(alpha=0.3)

            # ---------------- Regression Plot ----------------

            # Blue random points
            axes[1].scatter(
                x1,
                y_true,
                color="blue",
                s=20,
                alpha=0.7,
                label="True Data"
            )

            # Straight prediction line
            x2_fixed = 2.5
            line_inputs = dlf.Tensor([
                [float(x), x2_fixed] for x in x_line
            ])

            line_pred = np.array(
                model.forward(line_inputs).data()
            ).flatten()

            axes[1].plot(
                x_line,
                line_pred,
                color="red",
                linewidth=3,
                label="Prediction"
            )

            axes[1].set_xlim(0, 5)
            axes[1].set_ylim(-5, 30)
            axes[1].set_xticks(np.arange(0, 6, 1))
            axes[1].set_yticks(np.arange(-5, 31, 1))

            axes[1].set_xlabel("x₁")
            axes[1].set_ylabel("y")
            axes[1].set_title(f"Regression (Epoch {epoch})")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)


    plt.ioff()

    # Final prediction
    model.eval()

    plt.show()


main()