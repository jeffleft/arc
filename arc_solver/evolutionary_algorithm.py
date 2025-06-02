import random
import json
import os
import copy
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import csv
import hashlib

# --- File Loading Utilities ---

def load_json_file(file_path: str) -> Optional[List[Dict]]:
    """Loads a JSON file and returns its content (list of dicts)."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

def load_text_file(file_path: str) -> Optional[str]:
    """Loads a text file and returns its content as a string."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

# --- Individual Class ---

class Individual:
    """
    Represents an individual in the evolutionary algorithm.
    An individual consists of a prompt and a set of tool definitions.
    """
    def __init__(self, prompt_text: str, tool_definitions: List[Dict]):
        """
        Initializes an Individual.

        Args:
            prompt_text (str): The text of the prompt.
            tool_definitions (List[Dict]): A list of tool definition dictionaries.
        """
        self.prompt_text: str = prompt_text
        self.tool_definitions: List[Dict] = tool_definitions
        self.fitness: Optional[float] = None # Overall fitness score
        self.fitness_details: Dict[str, Any] = {} # Detailed breakdown (score, success_rate, tokens, etc.)


    def __repr__(self) -> str:
        """
        Returns a string representation of the Individual.
        """
        return (f"Individual(fitness={self.fitness:.4f}, "
                f"prompt_len={len(self.prompt_text)}, "
                f"num_tools={len(self.tool_definitions)})")

    def clone(self) -> 'Individual':
        """
        Creates a deep copy of this Individual.
        """
        cloned = Individual(
            prompt_text=self.prompt_text,
            tool_definitions=copy.deepcopy(self.tool_definitions)
        )
        cloned.fitness = self.fitness
        cloned.fitness_details = copy.deepcopy(self.fitness_details)
        return cloned

    def get_hash(self) -> str:
        """
        Generates a unique SHA256 hash for the individual based on its
        prompt text and tool definitions.
        """
        # Hash prompt text
        prompt_hash = hashlib.sha256(self.prompt_text.encode('utf-8')).hexdigest()

        # Canonical representation and hash for tool definitions
        # 1. Sort list of tool dictionaries by name (if available, else by first key for consistency)
        # 2. For each dictionary, create a JSON string with sorted keys.
        try:
            # Attempt to sort by tool name, assuming 'name' key exists.
            # Fallback for tools without a 'name' or if it's not consistently present.
            sorted_tools = sorted(self.tool_definitions, key=lambda x: x.get('name', str(x)))
        except TypeError: # Handles cases where tools might not all have comparable 'name' fields (e.g. mixing None with str)
            # Fallback to sorting by the string representation of the tool definition
             sorted_tools = sorted(self.tool_definitions, key=lambda x: json.dumps(x, sort_keys=True))


        # Serialize the now sorted list of (hopefully consistently structured) tools
        # with sorted keys within each tool definition.
        tools_json_string = json.dumps(sorted_tools, sort_keys=True)
        tools_hash = hashlib.sha256(tools_json_string.encode('utf-8')).hexdigest()

        # Combine prompt and tools hash
        combined_hash_input = f"{prompt_hash}-{tools_hash}"
        final_hash = hashlib.sha256(combined_hash_input.encode('utf-8')).hexdigest()
        return final_hash

# --- Historical Data Management ---

class HistoricalDataManager:
    """
    Manages a cache of previously evaluated individuals and their fitness results
    to avoid re-evaluating known individuals on the same set of tasks.
    It can also parse results from previous EA run directories.
    """
    # Placeholder for a hash representing a standard set of tasks used in EA runs.
    # This should ideally be generated from the actual list of task files used in those runs.
    # For now, we use a fixed string. If different EA runs use different task sets for evaluation,
    # this part needs to be more dynamic (e.g., store task_set_hash in gen_XXX/fitness_details.json).
    STANDARD_EA_TASK_SET_NAME = "StandardArcTaskSet_Set1_10Tasks"

    def __init__(self, cache_file_path: str):
        self.cache_file_path: str = cache_file_path
        self.results_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (individual_hash, task_set_hash) -> fitness_details
        self.load_cache_from_file()
        self.standard_task_set_hash_for_ea_parsing = self._generate_task_set_hash([self.STANDARD_EA_TASK_SET_NAME])


    def load_cache_from_file(self) -> None:
        """Loads the results cache from a JSON file."""
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, 'r') as f:
                    # JSON keys must be strings, so tuples were converted. Convert back.
                    loaded_data = json.load(f)
                    self.results_cache = {
                        (eval(key_str)[0], eval(key_str)[1]): value
                        for key_str, value in loaded_data.items()
                    }
                print(f"Loaded {len(self.results_cache)} historical results from {self.cache_file_path}")
            except (FileNotFoundError, json.JSONDecodeError, TypeError, SyntaxError) as e:
                print(f"Error loading cache from {self.cache_file_path}: {e}. Starting with an empty cache.")
                self.results_cache = {}
        else:
            print(f"Cache file {self.cache_file_path} not found. Starting with an empty cache.")
            self.results_cache = {}

    def save_cache_to_file(self) -> None:
        """Saves the current results cache to a JSON file."""
        try:
            # Convert tuple keys to string representations for JSON compatibility
            serializable_cache = {str(key_tuple): value for key_tuple, value in self.results_cache.items()}
            with open(self.cache_file_path, 'w') as f:
                json.dump(serializable_cache, f, indent=2)
            # print(f"Saved {len(self.results_cache)} historical results to {self.cache_file_path}")
        except IOError as e:
            print(f"Error saving cache to {self.cache_file_path}: {e}")

    def get_cached_result(self, individual_hash: str, task_set_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached fitness details for an individual on a specific task set.
        """
        return self.results_cache.get((individual_hash, task_set_hash))

    def store_result(self, individual_hash: str, task_set_hash: str, fitness_details: Dict[str, Any]) -> None:
        """
        Stores fitness details for an individual on a task set and saves the cache.
        """
        self.results_cache[(individual_hash, task_set_hash)] = fitness_details
        self.save_cache_to_file() # Save after each new result for persistence

    def _generate_task_set_hash(self, task_filenames: List[str]) -> str:
        """
        Generates a SHA256 hash for a sorted list of task filenames.
        """
        if not task_filenames:
            return hashlib.sha256("EMPTY_TASK_SET".encode('utf-8')).hexdigest()
        sorted_filenames = sorted(task_filenames)
        combined_string = ";".join(sorted_filenames)
        return hashlib.sha256(combined_string.encode('utf-8')).hexdigest()

    def parse_ea_run_directory(self, run_dir_path: str) -> int:
        """
        Parses results from a single EA run directory (e.g., 'arc_solver/ea_runs/run_YYYYMMDD_HHMMSS/').
        It loads prompt, tools, and fitness details from each 'gen_XXX' subdirectory.

        Args:
            run_dir_path (str): Path to the specific EA run directory.

        Returns:
            int: Number of historical records loaded from this run directory.
        """
        loaded_count = 0
        if not os.path.isdir(run_dir_path):
            print(f"Warning: EA run directory not found: {run_dir_path}")
            return 0

        for gen_dir_name in sorted(os.listdir(run_dir_path)): # Ensure order for reproducibility if needed
            if gen_dir_name.startswith("gen_") and os.path.isdir(os.path.join(run_dir_path, gen_dir_name)):
                gen_path = os.path.join(run_dir_path, gen_dir_name)

                prompt_file = os.path.join(gen_path, "prompt.txt")
                tools_file = os.path.join(gen_path, "tools.json")
                fitness_file = os.path.join(gen_path, "fitness_details.json")

                if os.path.exists(prompt_file) and os.path.exists(tools_file) and os.path.exists(fitness_file):
                    prompt_text = load_text_file(prompt_file)
                    tool_definitions = load_json_file(tools_file)
                    fitness_details_from_file = load_json_file(fitness_file)

                    if prompt_text is not None and tool_definitions is not None and fitness_details_from_file is not None:
                        # Create a temporary Individual just for hashing, or use components directly
                        # This assumes the structure of fitness_details_from_file is consistent
                        # with what `calculate_fitness` would store (e.g., contains 'overall_score').
                        temp_individual = Individual(prompt_text, tool_definitions)
                        individual_hash = temp_individual.get_hash()

                        # For now, assume these results are for the 'STANDARD_EA_TASK_SET_NAME'
                        # A more robust system might store the actual task_set_hash in fitness_details.json
                        task_set_hash_for_parsed_run = self.standard_task_set_hash_for_ea_parsing

                        if self.get_cached_result(individual_hash, task_set_hash_for_parsed_run) is None:
                            self.store_result(individual_hash, task_set_hash_for_parsed_run, fitness_details_from_file)
                            loaded_count += 1
                        # else:
                            # print(f"Skipping already cached result for individual from {gen_path}")
                # else:
                    # print(f"Skipping {gen_path} as some files are missing.")
        print(f"Loaded {loaded_count} new historical records from EA run directory: {run_dir_path}")
        return loaded_count

    def load_archived_runs(self, archive_root_dirs: List[str]) -> None:
        """
        Loads historical data from archived EA run directories.

        Args:
            archive_root_dirs (List[str]): A list of root directories where EA runs are stored
                                          (e.g., ['arc_solver/ea_runs']).
        """
        total_loaded = 0
        for root_dir in archive_root_dirs:
            if not os.path.isdir(root_dir):
                print(f"Warning: Archive root directory not found: {root_dir}")
                continue

            for item_name in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item_name)
                # Heuristic: EA run directories start with "run_"
                if item_name.startswith("run_") and os.path.isdir(item_path):
                    total_loaded += self.parse_ea_run_directory(item_path)

        if total_loaded > 0:
            print(f"Finished loading archives. Total new records added to cache: {total_loaded}")
            self.save_cache_to_file() # Save once after all parsing

# --- EA Core Functions ---

def initialize_population(pop_size: int, initial_prompt_file: str, initial_tools_file: str) -> List[Individual]:
    """
    Initializes a population of Individuals.

    The first individual is created directly from the base prompt and tools.
    Subsequent individuals are created by making small random modifications
    to this base individual.

    Args:
        pop_size (int): The size of the population to create.
        initial_prompt_file (str): Path to the file containing the initial prompt text.
        initial_tools_file (str): Path to the JSON file containing initial tool definitions.

    Returns:
        List[Individual]: A list of initialized Individual objects.
    """
    population = []
    base_prompt_text = load_text_file(initial_prompt_file)
    base_tool_definitions = load_json_file(initial_tools_file)

    if base_prompt_text is None:
        base_prompt_text = "Solve the ARC puzzle by analyzing examples and applying tools." # Fallback
        print(f"Warning: Using fallback initial prompt due to loading error from {initial_prompt_file}")
    if base_tool_definitions is None:
        base_tool_definitions = [] # Fallback
        print(f"Warning: Using fallback empty toolset due to loading error from {initial_tools_file}")

    # Create the base individual
    base_individual = Individual(base_prompt_text, base_tool_definitions)
    population.append(base_individual)

    # Create variations for the rest of the population
    for _ in range(pop_size - 1):
        new_prompt = base_prompt_text
        new_tools = copy.deepcopy(base_tool_definitions)

        # Placeholder: Small random modification to prompt
        if random.random() < 0.5: # 50% chance to modify prompt
            new_prompt += " " + random.choice(['Think step-by-step.', 'Focus on patterns.', 'Be concise.'])

        # Placeholder: Small random modification to tools (e.g., remove one tool)
        if new_tools and random.random() < 0.3 and len(new_tools) > 1: # 30% chance to remove a tool
            tool_to_remove = random.choice(new_tools)
            new_tools.remove(tool_to_remove)

        population.append(Individual(new_prompt, new_tools))

    return population

def calculate_fitness(individual: Individual, tasks_to_evaluate: List[str]) -> float:
    """
    Calculates the fitness of an individual.
    Placeholder: Fitness is based on prompt length and number of tools.
    This will be replaced by actual solver runs on ARC tasks.

    Args:
        individual (Individual): The individual to evaluate.
        tasks_to_evaluate (List[str]): A list of task filenames (currently unused by placeholder).

    Returns:
        float: The calculated fitness score.
    """
    # Temporary placeholder logic: simple heuristic
    # A more complex fitness might involve running the solver with the individual's
    # prompt and tools against a subset of ARC tasks and scoring based on success/efficiency.

    # For now, use the placeholder heuristic
    fitness_score = (len(individual.prompt_text) * 0.01) + (len(individual.tool_definitions) * 0.1) # Adjusted scale
    fitness_score += random.uniform(0, 1) # Add some noise to make fitness values less uniform for testing

    # This is where the actual results from the solver would be structured.
    # For the placeholder, we create a compatible structure.
    actual_fitness_details = {
        'overall_score': fitness_score, # This is the primary fitness value EA will use
        'success_rate': 0.0, # Placeholder
        'average_tokens': 0,   # Placeholder
        'tasks_attempted': len(tasks_to_evaluate), # Example of other metrics
        'notes': "Fitness calculated using placeholder heuristic."
    }

    # Store this result (even if from heuristic) in the cache
    historical_manager.store_result(individual_hash, task_set_hash, actual_fitness_details)

    individual.fitness = fitness_score
    individual.fitness_details = actual_fitness_details # Store detailed breakdown
    return fitness_score

def select_parents(population: List[Individual], tournament_size: int) -> Individual:
    """
    Selects a parent from the population using tournament selection.

    Args:
        population (List[Individual]): The current population.
        tournament_size (int): The number of individuals to select for the tournament.

    Returns:
        Individual: The fittest individual from the tournament.
    """
    if not population:
        raise ValueError("Population cannot be empty for parent selection.")
    if tournament_size <= 0:
        tournament_size = 2 # Default to a small tournament

    # Ensure tournament size is not larger than population size
    actual_tournament_size = min(tournament_size, len(population))

    tournament_contenders = random.sample(population, actual_tournament_size)

    # Assumes fitness has been calculated and higher is better
    winner = tournament_contenders[0]
    for contender in tournament_contenders[1:]:
        if contender.fitness is not None and (winner.fitness is None or contender.fitness > winner.fitness):
            winner = contender
    return winner

def apply_crossover(parent1: Individual, parent2: Individual) -> List[Individual]:
    """
    Applies crossover between two parent Individuals to produce offspring.
    Placeholder: Simple swap of prompt/tools.

    Args:
        parent1 (Individual): The first parent.
        parent2 (Individual): The second parent.

    Returns:
        List[Individual]: A list containing two new offspring Individuals.
    """
    # Ensure parents are distinct for meaningful crossover, though not strictly necessary for this placeholder
    # if parent1 is parent2:
    #     return [parent1.clone(), parent2.clone()] # Or handle as an error/special case

    # Placeholder: Offspring1 gets prompt from parent1, tools from parent2.
    # Offspring2 gets prompt from parent2, tools from parent1.
    offspring1_prompt = parent1.prompt_text
    offspring1_tools = copy.deepcopy(parent2.tool_definitions)
    offspring1 = Individual(offspring1_prompt, offspring1_tools)

    offspring2_prompt = parent2.prompt_text
    offspring2_tools = copy.deepcopy(parent1.tool_definitions)
    offspring2 = Individual(offspring2_prompt, offspring2_tools)

    return [offspring1, offspring2]

def apply_mutation(individual: Individual, mutation_rate_prompt: float, mutation_rate_tools: float) -> None:
    """
    Applies mutation to an Individual's prompt and/or tool definitions.
    Modifies the individual in-place.

    Args:
        individual (Individual): The individual to mutate.
        mutation_rate_prompt (float): The probability of mutating the prompt.
        mutation_rate_tools (float): The probability of mutating the tool definitions.
    """
    # Mutate prompt
    if random.random() < mutation_rate_prompt:
        # Placeholder: Append a random character or a short phrase
        mutation_type = random.choice(["append_char", "append_phrase", "change_word"])
        if mutation_type == "append_char" and individual.prompt_text:
            individual.prompt_text += random.choice("abcdefghijklmnopqrstuvwxyz.!? ")
        elif mutation_type == "append_phrase":
            individual.prompt_text += " " + random.choice(["Consider alternatives.", "Verify carefully.", "Seek efficiency."])
        # More sophisticated mutations could involve NLP techniques, e.g., synonym replacement, rephrasing parts.
        # print(f"Mutated prompt for individual (ID: {id(individual)})")


    # Mutate tool definitions
    if random.random() < mutation_rate_tools and len(individual.tool_definitions) > 0:
        mutation_type = random.choice(["remove_tool", "add_placeholder_tool", "modify_tool_desc"])

        if mutation_type == "remove_tool" and len(individual.tool_definitions) > 1:
            # Randomly remove one tool, if more than one exists
            tool_to_remove = random.choice(individual.tool_definitions)
            individual.tool_definitions.remove(tool_to_remove)
            # print(f"Removed tool from individual (ID: {id(individual)})")
        elif mutation_type == "add_placeholder_tool":
            # Add a new placeholder tool (more complex tools would require careful construction)
            # This is a very basic example. Real tool addition would be complex.
            new_placeholder_tool = {
                "type": "function",
                "name": f"placeholder_tool_{random.randint(100,999)}",
                "description": "A new experimental tool.",
                "parameters": {"type": "object", "properties": {"rationale": {"type": "string"}}, "required": ["rationale"]}
            }
            # Ensure not too many tools are added
            if len(individual.tool_definitions) < 15: # Arbitrary limit
                 individual.tool_definitions.append(new_placeholder_tool)
            # print(f"Added placeholder tool to individual (ID: {id(individual)})")
        elif mutation_type == "modify_tool_desc" and individual.tool_definitions:
            # Modify a description of a random tool
            tool_to_modify = random.choice(individual.tool_definitions)
            if "description" in tool_to_modify:
                 tool_to_modify["description"] += " (mutated)"
            # print(f"Modified tool description for individual (ID: {id(individual)})")
    # Note: Fitness is reset to None as the individual has changed.
    individual.fitness = None
    individual.fitness_details = {}


# --- Main Evolutionary Algorithm Loop ---

def run_evolution(
    generations: int,
    pop_size: int,
    mutation_rate_prompt: float,
    mutation_rate_tools: float,
    crossover_rate: float,
    tournament_size: int,
    initial_prompt_file: str,
    initial_tools_file: str,
    tasks_to_evaluate: List[str],
    elitism_count: int = 1,
    historical_data_manager: Optional[HistoricalDataManager] = None # Added
    ) -> Optional[Individual]: # Return type can be Optional[Individual]
    """
    Runs the evolutionary algorithm.

    Args:
        generations (int): Number of generations to run.
        pop_size (int): Size of the population.
        mutation_rate_prompt (float): Probability of prompt mutation.
        mutation_rate_tools (float): Probability of toolset mutation.
        crossover_rate (float): Probability of crossover.
        tournament_size (int): Size of the selection tournament.
        initial_prompt_file (str): Path to the initial prompt file.
        initial_tools_file (str): Path to the initial tools JSON file.
        tasks_to_evaluate (List[str]): List of task filenames for fitness evaluation.
        elitism_count (int): Number of best individuals to carry over to the next generation.
        historical_data_manager (HistoricalDataManager, optional): Manager for historical results.

    Returns:
        Optional[Individual]: The best individual found after all generations, or None if error.
    """
    if historical_data_manager is None:
        # Default cache path if no manager is provided (e.g. for direct script runs)
        # SCRIPT_DIR needs to be defined if this is used, typically os.path.dirname(__file__)
        # For robustness, ensure SCRIPT_DIR is available or handle its absence.
        try:
            script_dir_for_cache = os.path.dirname(__file__)
        except NameError: # __file__ not defined (e.g. in some interactive environments)
            script_dir_for_cache = "." # Default to current directory
        default_cache_path = os.path.join(script_dir_for_cache, "ea_runs", "results_cache.json")
        os.makedirs(os.path.dirname(default_cache_path), exist_ok=True)
        historical_data_manager = HistoricalDataManager(cache_file_path=default_cache_path)
        # Attempt to load any existing archived runs if running standalone
        historical_data_manager.load_archived_runs([os.path.join(script_dir_for_cache, "ea_runs")])


    print("Initializing population...")
    population = initialize_population(pop_size, initial_prompt_file, initial_tools_file)

    # --- Run Directory Creation ---
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # SCRIPT_DIR should be defined if this part is reached via __main__
    # If called as a function, os.path.dirname(__file__) might not be what's expected if imported.
    # For now, assume __file__ is contextually correct or SCRIPT_DIR is passed/global.
    try:
        ea_runs_base_path = os.path.join(os.path.dirname(__file__), "ea_runs")
    except NameError:
         ea_runs_base_path = "ea_runs" # Fallback if __file__ is not defined
    os.makedirs(ea_runs_base_path, exist_ok=True)
    current_run_dir = os.path.join(ea_runs_base_path, f"run_{run_timestamp}")
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"EA outputs will be saved to: {current_run_dir}")

    # --- CSV Log File Setup ---
    log_file_path = os.path.join(current_run_dir, "evolution_log.csv")
    try:
        csv_file = open(log_file_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        header = ["generation", "timestamp", "best_fitness", "average_fitness", "worst_fitness",
                  "best_prompt_file", "best_tools_file", "best_fitness_details_file"]
        csv_writer.writerow(header)
    except IOError as e:
        print(f"Fatal: Could not open CSV log file at {log_file_path}: {e}")
        return None # Cannot proceed without logging

    overall_best_individual: Optional[Individual] = None

    for gen in range(generations):
        generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        print(f"\n--- Generation {gen + 1}/{generations} (Timestamp: {generation_timestamp}) ---")

        # Calculate fitness for each individual
        print("Calculating fitness...")
        task_set_hash = historical_data_manager._generate_task_set_hash(tasks_to_evaluate)
        for i, individual in enumerate(population):
            if individual.fitness is None: # Only calculate if not already set
                individual_hash = individual.get_hash()
                cached_fitness_details = historical_data_manager.get_cached_result(individual_hash, task_set_hash)
                if cached_fitness_details:
                    individual.fitness = cached_fitness_details.get('overall_score') # Ensure key exists
                    individual.fitness_details = cached_fitness_details
                    print(f"  Individual {i} (Hash: {individual_hash[:8]}...): Fitness from CACHE = {individual.fitness:.4f}")
                else:
                    # Placeholder for actual solver call. For now, uses heuristic.
                    calculate_fitness(individual, tasks_to_evaluate, historical_data_manager) # Pass manager
                    print(f"  Individual {i} (Hash: {individual_hash[:8]}...): Fitness CALCULATED = {individual.fitness:.4f}")
            # else:
                # print(f"  Individual {i}: Fitness already known = {individual.fitness:.4f}")

        # Sort population by fitness (descending, higher is better)
        population.sort(key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf'), reverse=True)

        # --- Fitness Statistics ---
        current_best_individual_this_gen = population[0] if population else None
        best_fitness_this_gen = current_best_individual_this_gen.fitness if current_best_individual_this_gen and current_best_individual_this_gen.fitness is not None else -float('inf')
        worst_fitness_this_gen = population[-1].fitness if population and population[-1].fitness is not None else -float('inf')

        valid_fitness_scores = [ind.fitness for ind in population if ind.fitness is not None]
        average_fitness_this_gen = sum(valid_fitness_scores) / len(valid_fitness_scores) if valid_fitness_scores else 0.0

        # Update overall best individual
        if current_best_individual_this_gen and (overall_best_individual is None or best_fitness_this_gen > (overall_best_individual.fitness or -float('inf'))):
            overall_best_individual = current_best_individual_this_gen.clone()
            print(f"New overall best individual found: {overall_best_individual}")

        # --- Console Logging ---
        print(f"Best Fitness in Gen {gen + 1}: {best_fitness_this_gen:.4f}")
        print(f"Average Fitness in Gen {gen + 1}: {average_fitness_this_gen:.4f}")
        print(f"Worst Fitness in Gen {gen + 1}: {worst_fitness_this_gen:.4f}")
        # print(f"Overall Best Individual so far: {overall_best_individual}") # Can be verbose

        # --- Save Best of Generation ---
        gen_dir_name = f"gen_{gen + 1:03d}" # Padded generation number
        generation_dir_path = os.path.join(current_run_dir, gen_dir_name)
        os.makedirs(generation_dir_path, exist_ok=True)

        best_prompt_file_rel = ""
        best_tools_file_rel = ""
        best_fitness_file_rel = ""

        if current_best_individual_this_gen:
            prompt_file_path = os.path.join(generation_dir_path, "prompt.txt")
            tools_file_path = os.path.join(generation_dir_path, "tools.json")
            fitness_file_path = os.path.join(generation_dir_path, "fitness_details.json")

            try:
                with open(prompt_file_path, 'w') as f:
                    f.write(current_best_individual_this_gen.prompt_text)
                best_prompt_file_rel = os.path.join(gen_dir_name, "prompt.txt")
            except IOError as e:
                print(f"Error saving prompt for gen {gen+1}: {e}")

            try:
                with open(tools_file_path, 'w') as f:
                    json.dump(current_best_individual_this_gen.tool_definitions, f, indent=2)
                best_tools_file_rel = os.path.join(gen_dir_name, "tools.json")
            except IOError as e:
                print(f"Error saving tools for gen {gen+1}: {e}")

            try:
                fitness_details = {"fitness": current_best_individual_this_gen.fitness}
                # Future: Add more details like {"score": ..., "complexity_penalty": ...}
                with open(fitness_file_path, 'w') as f:
                    json.dump(fitness_details, f, indent=2)
                best_fitness_file_rel = os.path.join(gen_dir_name, "fitness_details.json")
            except IOError as e:
                print(f"Error saving fitness details for gen {gen+1}: {e}")

        # --- CSV Logging ---
        csv_writer.writerow([
            gen + 1,
            generation_timestamp,
            f"{best_fitness_this_gen:.4f}",
            f"{average_fitness_this_gen:.4f}",
            f"{worst_fitness_this_gen:.4f}",
            best_prompt_file_rel,
            best_tools_file_rel,
            best_fitness_file_rel
        ])
        csv_file.flush() # Ensure data is written to disk periodically

        # --- Create the next generation ---
        next_generation: List[Individual] = []

        # Elitism: carry over the best N individuals
        if elitism_count > 0 and elitism_count <= pop_size:
            next_generation.extend([ind.clone() for ind in population[:elitism_count]])
            # print(f"Carried over {len(next_generation)} elites.")

        # Fill the rest of the next generation using selection, crossover, and mutation
        print("Generating next generation...")
        while len(next_generation) < pop_size:
            parent1 = select_parents(population, tournament_size)
            parent2 = select_parents(population, tournament_size)

            # Ensure two distinct parents for crossover if possible, though simple crossover might not care
            # attempts = 0
            # while parent1 is parent2 and len(population) > 1 and attempts < 10 : # Avoid infinite loop for tiny pops
            #     parent2 = select_parents(population, tournament_size)
            #     attempts += 1

            offspring_list = []
            if random.random() < crossover_rate:
                offspring_list.extend(apply_crossover(parent1, parent2))
            else:
                # No crossover, parents' clones move to offspring list (for mutation)
                offspring_list.append(parent1.clone())
                if len(next_generation) + 1 < pop_size: # Ensure space for a second offspring if no crossover
                    offspring_list.append(parent2.clone())

            for offspring in offspring_list:
                if len(next_generation) < pop_size:
                    apply_mutation(offspring, mutation_rate_prompt, mutation_rate_tools)
                    next_generation.append(offspring)
                else:
                    break # Next generation is full

        population = next_generation
        # print(f"Population size for next gen: {len(population)}")

    # After all generations, calculate fitness for the final population one last time
    print("\n--- Final Population Fitness Calculation ---")
    for individual in population:
        if individual.fitness is None:
             calculate_fitness(individual, tasks_to_evaluate)

    population.sort(key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf'), reverse=True)

    # Use the overall_best_individual tracked across all generations
    final_best_individual_to_return = overall_best_individual if overall_best_individual else (population[0] if population else None)

    print(f"\nEvolution Finished. Overall Best individual found: {final_best_individual_to_return}")

    # --- Save Overall Best Individual ---
    if final_best_individual_to_return:
        print(f"  Prompt: '{final_best_individual_to_return.prompt_text[:100]}...'")
        print(f"  Tool Count: {len(final_best_individual_to_return.tool_definitions)}")

        try:
            with open(os.path.join(current_run_dir, "overall_best_prompt.txt"), 'w') as f:
                f.write(final_best_individual_to_return.prompt_text)
            with open(os.path.join(current_run_dir, "overall_best_tools.json"), 'w') as f:
                json.dump(final_best_individual_to_return.tool_definitions, f, indent=2)
            with open(os.path.join(current_run_dir, "overall_best_fitness.json"), 'w') as f:
                json.dump({"fitness": final_best_individual_to_return.fitness}, f, indent=2)
            print(f"Overall best individual saved in {current_run_dir}")
        except IOError as e:
            print(f"Error saving overall best individual: {e}")

    csv_file.close() # Close the CSV log file
    return final_best_individual_to_return


# --- Main Execution Block ---

if __name__ == '__main__':
    print("Starting Evolutionary Algorithm for ARC Solver...")

    # Configuration for the EA
    GENERATIONS = 10 # Number of generations
    POPULATION_SIZE = 20 # Population size
    MUTATION_RATE_PROMPT = 0.15 # Probability of mutating an individual's prompt
    MUTATION_RATE_TOOLS = 0.1 # Probability of mutating an individual's toolset
    CROSSOVER_RATE = 0.7 # Probability of performing crossover
    TOURNAMENT_SIZE = 3  # Size of the tournament for parent selection
    ELITISM_COUNT = 2    # Number of best individuals to carry to next generation

    # Define base path for prompts and tools relative to this script's location
    # Assuming this script is in arc_solver/
    SCRIPT_DIR = os.path.dirname(__file__)
    INITIAL_PROMPT_FILE = os.path.join(SCRIPT_DIR, "prompts", "initial_prompt.txt")
    INITIAL_TOOLS_FILE = os.path.join(SCRIPT_DIR, "tools", "tool_definitions.json")

    # Placeholder for tasks to evaluate against. In a real scenario, these would be
    # actual ARC task file paths or identifiers used by a sophisticated fitness function.
    # For now, this list is not directly used by the placeholder `calculate_fitness`.
    TASKS_TO_EVALUATE = ["task1.json", "task2.json"] # Example task names

    # Run the evolutionary algorithm
    best_solution = run_evolution(
        generations=GENERATIONS,
        pop_size=POPULATION_SIZE,
        mutation_rate_prompt=MUTATION_RATE_PROMPT,
        mutation_rate_tools=MUTATION_RATE_TOOLS,
        crossover_rate=CROSSOVER_RATE,
        tournament_size=TOURNAMENT_SIZE,
        initial_prompt_file=INITIAL_PROMPT_FILE,
        initial_tools_file=INITIAL_TOOLS_FILE,
        tasks_to_evaluate=TASKS_TO_EVALUATE,
        elitism_count=ELITISM_COUNT
    )

    print("\n--- Best Solution Details ---")
    if best_solution:
        print(f"Fitness: {best_solution.fitness:.4f}")
        print(f"Prompt Text:\n{best_solution.prompt_text}")
        print(f"\nTool Definitions ({len(best_solution.tool_definitions)} tools):")
        for i, tool_def in enumerate(best_solution.tool_definitions):
            print(f"  Tool {i+1}: {tool_def.get('name', 'N/A')} - {tool_def.get('description', 'N/A')[:50]}...")
    else:
        print("No solution found or population was empty.")

    print("\nEvolutionary Algorithm run complete.")
