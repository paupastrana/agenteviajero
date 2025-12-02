import random
import math
import json
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            # Procesar la solicitud
            result = self.evolve_tsp(data)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    # ---------- TU LÓGICA DEL ALGORITMO GENÉTICO ----------
    
    def calculate_distance(self, city1, city2):
        return math.sqrt((city1['x'] - city2['x'])**2 + (city1['y'] - city2['y'])**2)
    
    def calculate_total_route_distance(self, route, cities, city_map):
        if len(cities) < 1 or len(route) < 1:
            return float('inf')
        
        distance = 0.0
        base_city = cities[0]
        
        # Distancia desde ciudad base a primera ciudad de la ruta
        first_city = city_map.get(route[0])
        if first_city:
            distance += self.calculate_distance(base_city, first_city)
        else:
            return float('inf')
        
        # Distancias entre ciudades de la ruta
        for i in range(len(route) - 1):
            city_a = city_map.get(route[i])
            city_b = city_map.get(route[i+1])
            if city_a and city_b:
                distance += self.calculate_distance(city_a, city_b)
            else:
                return float('inf')
        
        # Regresar a ciudad base
        end_city = city_map.get(route[-1])
        if end_city:
            distance += self.calculate_distance(end_city, base_city)
        
        return distance
    
    def initialize_population(self, cities, size):
        base_city_id = cities[0]['id']
        city_ids_to_permute = [city['id'] for city in cities if city['id'] != base_city_id]
        
        population = []
        for _ in range(size):
            chromosome = city_ids_to_permute[:]
            random.shuffle(chromosome)
            population.append(chromosome)
        return population
    
    def fitness(self, distance):
        return 1.0 / (distance + 1e-6)
    
    def tournament_selection(self, population, cities, city_map, k=3):
        if len(population) < k:
            k = len(population)
        
        tournament_pool = random.sample(population, k)
        best_chromosome = None
        best_fitness = -1
        
        for chromosome in tournament_pool:
            dist = self.calculate_total_route_distance(chromosome, cities, city_map)
            current_fitness = self.fitness(dist)
            
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_chromosome = chromosome
        
        return best_chromosome[:] if best_chromosome else random.choice(population)[:]
    
    def order_crossover(self, p1, p2):
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
    
    def swap_mutation(self, route, rate):
        if random.random() < rate and len(route) >= 2:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route
    
    def evolve_tsp(self, data):
        cities = data.get('cities', [])
        params = data.get('parameters', {})
        current_population = data.get('current_population', [])
        best_route = data.get('best_route', [])
        best_distance = data.get('best_distance', float('inf'))
        generation = data.get('generation', 0)
        
        # Parámetros por defecto
        population_size = params.get('population_size', 50)
        mutation_rate = params.get('mutation_rate', 0.01)
        generations_per_step = params.get('generations_per_step', 1)
        
        num_cities_to_permute = len(cities) - 1
        
        if num_cities_to_permute < 1:
            return {
                "generation": generation + generations_per_step,
                "new_best_route": [],
                "new_best_distance": 0.0,
                "next_population": []
            }
        
        city_map = {city['id']: city for city in cities}
        
        # Inicializar población si es necesario
        population = current_population
        if (not population or 
            len(population) != population_size or 
            (len(population[0]) != num_cities_to_permute if num_cities_to_permute > 0 else False)):
            population = self.initialize_population(cities, population_size)
            best_route = []
            best_distance = float('inf')
        
        current_generation = generation
        best_dist = best_distance if best_distance != 0.0 else float('inf')
        best_rt = best_route
        
        # Evaluar población inicial
        for chromosome in population:
            dist = self.calculate_total_route_distance(chromosome, cities, city_map)
            if dist < best_dist:
                best_dist = dist
                best_rt = chromosome
        
        # Evolución por generaciones
        for _ in range(generations_per_step):
            current_generation += 1
            new_population = []
            
            # Elitismo: mantener el mejor
            if best_rt:
                new_population.append(best_rt[:])
            
            # Generar nueva población
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, cities, city_map)
                parent2 = self.tournament_selection(population, cities, city_map)
                
                child = self.order_crossover(parent1, parent2)
                mutated_child = self.swap_mutation(child, mutation_rate)
                
                new_population.append(mutated_child)
            
            population = new_population
            
            # Actualizar mejor ruta
            for chromosome in population:
                dist = self.calculate_total_route_distance(chromosome, cities, city_map)
                if dist < best_dist:
                    best_dist = dist
                    best_rt = chromosome
        
        return {
            "generation": current_generation,
            "new_best_route": best_rt,
            "new_best_distance": best_dist,
            "next_population": population
        }