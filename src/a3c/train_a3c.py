from my_a3c import A3CAgent

def main():
    agent = A3CAgent(lr=1e-3, gamma=0.99, workers=4)  # 4 parallel workers
    print("Starting A3C training...")
    agent.train()
    agent.save("a3c_flappy.pth")
    print("Training complete! Model saved.")

if __name__ == "__main__":
    main()
