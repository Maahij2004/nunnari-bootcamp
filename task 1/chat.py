import ollama
import time

MODEL = "llama3.2:3b"

roles = {
    "1": {
        "name": "Python Tutor",
        "prompt": "You are a patient Python tutor who explains concepts clearly with examples."
    },
    "2": {
        "name": "Fitness Coach",
        "prompt": "You are a motivating fitness coach who gives simple, practical health advice."
    },
    "3": {
        "name": "Travel Guide",
        "prompt": "You are a friendly travel guide who suggests destinations, tips, and itineraries."
    }
}


def show_roles():
    print("\nAvailable Roles:")
    for key, role in roles.items():
        print(f"{key}. {role['name']}")


def choose_role():
    show_roles()
    choice = input("Pick a role (number): ").strip()
    return roles.get(choice)


def main():
    role = choose_role()
    if not role:
        print("Invalid choice. Exiting.")
        return

    messages = [{"role": "system", "content": role["prompt"]}]

    print(f"\nRole set: {role['name']}")
    print("Commands: 'switch' | 'quit' | 'roles' (add custom role)")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        elif user_input.lower() == "switch":
            role = choose_role()
            if not role:
                print("Invalid choice. Staying in current role.")
                continue

            messages = [{"role": "system", "content": role["prompt"]}]
            print(f"\nRole set: {role['name']}")

        elif user_input.lower() == "roles":
            name = input("Enter role name: ").strip()
            prompt = input("Enter system prompt: ").strip()

            new_id = str(len(roles) + 1)
            roles[new_id] = {"name": name, "prompt": prompt}

            print(f"Role '{name}' added!")

        else:
            messages.append({"role": "user", "content": user_input})

            start_time = time.time()

            response = ollama.chat(
                model=MODEL,
                messages=messages
            )

            end_time = time.time()

            reply = response["message"]["content"]

            print(f"\n{role['name']}: {reply}")
            print(f"⏱ Response time: {end_time - start_time:.2f}s")

            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()