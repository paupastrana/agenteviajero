import random
import math
import json
from typing import List, Tuple, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from collections import defaultdict

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class City(BaseModel):
    id: int
    x: float
    y: float

class GeneticParameters(BaseModel):
    population_size: int = 50
    mutation_rate: float = 0.01
    generations_per_step: int = 1 

class EvolutionInput(BaseModel):
    cities: List[City]
    parameters: GeneticParameters
    current_population: List[List[int]] = []
    best_route: List[int] = []
    best_distance: float = 0.0
    generation: int = 0

class EvolutionOutput(BaseModel):
    generation: int
    new_best_route: List[int]
    new_best_distance: float
    next_population: List[List[int]]

def calculate_distance(city1, city2):
    return math.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)

def calculate_total_route_distance(route, cities, city_map):
    if len(cities) < 1:
        return 0.0
    if len(route) < 1:
        return float('inf')
    
    distance = 0.0
    base_city = cities[0]

    first_city = city_map.get(route[0])
    distance += calculate_distance(base_city, first_city)

    for i in range(len(route) - 1):
        city_a = city_map.get(route[i])
        city_b = city_map.get(route[i+1])
        if city_a and city_b:
            distance += calculate_distance(city_a, city_b)
        else:
            return float('inf')

    end_city = city_map.get(route[-1])
    distance += calculate_distance(end_city, base_city)
    
    return distance

def initialize_population(cities, size):
    city_ids_to_permute = [city.id for city in cities if city.id != cities[0].id]
    
    population = []
    for _ in range(size):
        chromosome = city_ids_to_permute[:] 
        random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness(distance: float):
    return 1.0 / (distance + 1e-6)

def tournament_selection(population, cities, city_map, k=3):
    tournament_pool = random.sample(population, min(k, len(population)))
    
    best_chromosome = None
    best_fitness = -1
    
    for chromosome in tournament_pool:
        dist = calculate_total_route_distance(chromosome, cities, city_map)
        current_fitness = fitness(dist)
        
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_chromosome = chromosome
            
    return best_chromosome[:] if best_chromosome else random.choice(population)[:]

def order_crossover(p1, p2):
    size = len(p1)
    if size < 2:
        return p1[:]
        
    a, b = random.sample(range(size), 2)
    start, end = min(a, b), max(a, b)
    
    child = [None] * size
    child[start:end] = p1[start:end]
    
    parent2_slice = [item for item in p2 if item not in child[start:end]]
    
    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = parent2_slice[p2_idx]
            p2_idx += 1
            
    return child

def swap_mutation(route, rate):
    if random.random() < rate:
        size = len(route)
        if size < 2:
            return route

        idx1, idx2 = random.sample(range(size), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

@app.post("/api/solve", response_model=EvolutionOutput)
async def evolve_tsp(data: EvolutionInput):
    cities = data.cities
    params = data.parameters

    num_cities_to_permute = len(cities) - 1
    if num_cities_to_permute < 1:
        return EvolutionOutput(
            generation=data.generation + params.generations_per_step,
            new_best_route=[],
            new_best_distance=0.0,
            next_population=[]
        )

    city_map = {c.id: c for c in cities}
    
    population = data.current_population
    
    if not population or len(population) != params.population_size or (len(population[0]) != num_cities_to_permute if num_cities_to_permute > 0 else True):
        population = initialize_population(cities, params.population_size)
        data.best_route = []
        data.best_distance = float('inf')
    
    current_generation = data.generation
    
    best_dist = data.best_distance if data.best_distance != 0.0 else float('inf')
    best_route = data.best_route
    
    for chromosome in population:
        dist = calculate_total_route_distance(chromosome, cities, city_map)
        if dist < best_dist:
            best_dist = dist
            best_route = chromosome

    for _ in range(params.generations_per_step):
        current_generation += 1
        new_population = []
        
        if best_route:
            new_population.append(best_route[:]) 
        
        while len(new_population) < params.population_size:
            parent1 = tournament_selection(population, cities, city_map)
            parent2 = tournament_selection(population, cities, city_map)
            
            child = order_crossover(parent1, parent2)
            mutated_child = swap_mutation(child, params.mutation_rate)
            
            new_population.append(mutated_child)
            
        population = new_population 
        
        for chromosome in population:
            dist = calculate_total_route_distance(chromosome, cities, city_map)
            if dist < best_dist:
                best_dist = dist
                best_route = chromosome
        
    return EvolutionOutput(
        generation=current_generation,
        new_best_route=best_route,
        new_best_distance=best_dist,
        next_population=population
    )
