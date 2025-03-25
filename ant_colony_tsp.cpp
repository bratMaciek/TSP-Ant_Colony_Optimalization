#include <iostream>
#include <ctime>
#include <limits>
#include <vector>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <functional>
#include <chrono>
#include <cstdlib> 
#include <random>
#include <string>
#include <queue>
#include <sstream>
#include <vector>
#include <limits>

#define MAX_VERTICES 100

using namespace std;

class IncidencyMatrix {
public:
    int* matrix;
    int vertices;
    int edges;
    int density;
    bool* visited;

    IncidencyMatrix(int a, int b, int c) : vertices(a), density(b), matrix(nullptr) {
    }

    void deleteIncidencyMatrix() {
        delete[] matrix;
    }
    // Alokowanie pamięci dla grafu nieskierowanego
    void generateNotDirectedArray() {
        edges = density * vertices * (vertices - 1) / 200;
        matrix = new int[vertices * edges];
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < edges; j++) {
                *(matrix + i * edges + j) = 0;
            }
        }
    }
    // Alokowanie pamięci dla grafu skierowanego
    void generateDirectedArray() {
        edges = density * vertices * (vertices - 1) / 100;
        matrix = new int[vertices * edges];
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < edges; j++) {
                *(matrix + i * edges + j) = 0;
            }
        }
    }
    // Funkcja wypisująca graf w postaci macierzy incydencji 
    void printArray() const {
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < edges; j++) {
                printf("%6d ", *(matrix + i * edges + j));
            }
            cout << endl;
        }
    }
    // Generowanie losowych wartości w grafie nieskierowanym
    void fillNotDirectedArray() {
        srand(time(NULL));
        bool** connection = new bool* [vertices];
        for (int i = 0; i < vertices; i++) {
            connection[i] = new bool[vertices];
        }

        bool* visited = new bool[vertices];
        visited[0] = true;
        for (int i = 1; i < vertices; i++) {
            visited[i] = false;
        }

        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < vertices; j++) {
                connection[i][j] = false;
            }
        }

        int sourceVertex = 0, destVertex = 0;
        int value = 0;
        int minWeight = 1;
        int maxWeight = RAND_MAX;

        for (int i = 0; i < vertices - 1; i++) {
            while (visited[destVertex]) {
                destVertex = rand() % vertices;
            }

            value = minWeight + rand() % (maxWeight - minWeight + 1);

            *(matrix + destVertex * edges + i) = value;
            *(matrix + sourceVertex * edges + i) = value;

            connection[sourceVertex][destVertex] = true;
            connection[destVertex][sourceVertex] = true;

            sourceVertex = destVertex;
            visited[destVertex] = true;
        }

        delete[] visited;

        for (int i = vertices - 1; i < edges; ++i) {
            int sourceVertex = rand() % vertices;
            int destVertex = rand() % vertices;
            while (sourceVertex == destVertex || connection[destVertex][sourceVertex]) {
                destVertex = rand() % vertices;
                sourceVertex = rand() % vertices;
            }
            connection[destVertex][sourceVertex] = true;
            connection[sourceVertex][destVertex] = true;

            int value = minWeight + rand() % (maxWeight - minWeight + 1);

            *(matrix + sourceVertex * edges + i) = value;
            *(matrix + destVertex * edges + i) = value;
        }

        for (int i = 0; i < vertices; i++) {
            delete[] connection[i];
        }
        delete[] connection;
    }
    // Generowanie losowych wartości w grafie skierowanym
    void fillDirectedArray() {
        srand(time(NULL));
        bool** connection = new bool* [vertices];
        for (int i = 0; i < vertices; i++) {
            connection[i] = new bool[vertices];
        }

        bool* visited = new bool[vertices];
        visited[0] = true;
        for (int i = 1; i < vertices; i++) {
            visited[i] = false;
        }

        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < vertices; j++) {
                connection[i][j] = false;
            }
        }

        int sourceVertex = 0, destVertex = 0;
        int value = 0;
        int minWeight = 1;
        int maxWeight = numeric_limits<int>::max();

        for (int i = 0; i < vertices - 1; i++) {
            while (visited[destVertex]) {
                destVertex = rand() % vertices;
            }

            value = minWeight + rand() % (maxWeight - minWeight + 1);

            *(matrix + destVertex * edges + i) = value;
            *(matrix + sourceVertex * edges + i) = -value;

            // connection[sourceVertex][destVertex] = true;
            connection[destVertex][sourceVertex] = true;

            sourceVertex = destVertex;
            visited[destVertex] = true;
        }

        delete[] visited;

        for (int i = vertices - 1; i < edges; ++i) {
            int sourceVertex = rand() % vertices;
            int destVertex = rand() % vertices;
            while (sourceVertex == destVertex || connection[destVertex][sourceVertex]) {
                destVertex = rand() % vertices;
                sourceVertex = rand() % vertices;
            }
            connection[destVertex][sourceVertex] = true;
            // connection[sourceVertex][destVertex] = true;

            int value = minWeight + rand() % (maxWeight - minWeight + 1);

            *(matrix + sourceVertex * edges + i) = value;
            *(matrix + destVertex * edges + i) = -value;
        }

        for (int i = 0; i < vertices; i++) {
            delete[] connection[i];
        }
        delete[] connection;
    }
    // Funkcja wczytująca nieskierowany graf z pliku
    void readNotDirectedGraphFromFile(string filename) {
        ifstream file(filename);

        if (!file.is_open()) {
            cerr << "Nie można otworzyć pliku" << endl;
            return;
        }

        file >> vertices >> edges;

        // Dealokowanie istniejącej pamięci, jeśli wskaźnik nie jest nullptr
        if (matrix != nullptr) {
            delete[] matrix;
        }

        matrix = new int[vertices * edges];
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < edges; j++) {
                *(matrix + i * edges + j) = 0;
            }
        }

        int vertex1, vertex2, value;
        for (int i = 0; i < edges; i++) {
            file >> vertex1 >> vertex2 >> value;
            *(matrix + vertex1 * edges + i) = value;
            *(matrix + vertex2 * edges + i) = value;
        }

        file.close();
    }
    // Funkcja wczytująca skierowany graf z pliku    
    void readDirectedGraphFromFile(string filename) {
        ifstream file(filename);

        if (!file.is_open()) {
            cerr << "Nie mozna otworzyc pliku" << endl;
            return;
        }

        file >> vertices >> edges;

        // Dealokowanie istniejącej pamięci, jeśli wskaźnik nie jest nullptr
        if (matrix != nullptr) {
            delete[] matrix;
        }

        matrix = new int[vertices * edges];
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < edges; j++) {
                *(matrix + i * edges + j) = 0;
            }
        }

        int vertex1, vertex2, value;
        for (int i = 0; i < edges; i++) {
            file >> vertex1 >> vertex2 >> value;
            *(matrix + vertex1 * edges + i) = value;
            *(matrix + vertex2 * edges + i) = -value;
        }

        file.close();
    }
    // Funkcja przksztłcająca graf z tsplib
    void fillFromTSPLIB(string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("Nie mozna otworzyc pliku!");
        }

        std::string line;
        int dimension = 0;
        std::vector<int> edgeWeights;
        bool readingWeights = false;

        while (std::getline(file, line)) {
            // Usuwanie białych znaków
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);

            if (line.rfind("DIMENSION", 0) == 0) {
                dimension = std::stoi(line.substr(line.find(":") + 1));
                vertices = dimension;
            }
            else if (line.rfind("EDGE_WEIGHT_SECTION", 0) == 0) {
                readingWeights = true;
            }
            else if (line == "EOF") {
                break;
            }
            else if (readingWeights) {
                std::istringstream iss(line);
                int weight;
                while (iss >> weight) {
                    edgeWeights.push_back(weight);
                }
            }
        }

        file.close();

        // Alokowanie pamięci na macierz incydencji
        edges = dimension * (dimension - 1) / 2; // Liczba krawędzi dla grafu pełnego
        matrix = new int[vertices * edges](); // Inicjalizacja zerami

        // Wypełnianie macierzy incydencji
        int edgeIndex = 0;
        for (int i = 0; i < dimension; ++i) {
            for (int j = 0; j <= i; ++j) {
                if (i != j) { // Uwzględniamy tylko krawędzie
                    *(matrix + i * edges + edgeIndex) = edgeWeights[edgeIndex];
                    *(matrix + j * edges + edgeIndex) = edgeWeights[edgeIndex];
                    ++edgeIndex;
                }
            }
        }
    }
    // Funkcja, która tworzy tablice wierzchołków jako liste "odwiedzin"
    void isVisited() {
        visited = new bool[vertices];   // parametr określający czy wierzchołek został odwiedzony
        for (int i = 0; i < vertices; i++) visited[i] = false;
    }
    // Funkcja do obliczania kosztu cyklu
    int calculateCost(const vector<int>& path) {
        int cost = 0;
        int edgeIndex;

        for (int i = 0; i < path.size() - 1; ++i) {
            if (findEdge(path[i], path[i + 1], edgeIndex)) {
                cost += *(matrix + path[i] * edges + edgeIndex);
            }
            else {
                return -1;
            }
        }
        // Dodanie kosztu powrotu do początkowego wierzchołka
        if (findEdge(path.back(), path[0], edgeIndex)) {
            cost += *(matrix + path.back() * edges + edgeIndex);
        }
        else {
            return -1;
        }

        return cost;
    }
    // Funkcja generująca sąsiadów przez zamianę dwóch wierzchołków
    vector<vector<int>> generateNeighbors(const vector<int>& path) {
        vector<vector<int>> neighbors;
        for (int i = 1; i < path.size() - 1; ++i) {
            for (int j = i + 1; j < path.size(); ++j) {
                vector<int> neighbor = path;
                swap(neighbor[i], neighbor[j]);
                neighbors.push_back(neighbor);
            }
        }
        return neighbors;
    }
    // Algorytm mrówkowy
    void antColonyOptimization(vector<int>& bestPath, int& bestCost, int numAnts, int maxIterations, double alpha, double beta, double evaporationRate, const string& strategy) {
        // Macierz feromonów, inicjalizowana wartością 1.0
        vector<vector<double>> pheromones(vertices, vector<double>(vertices, 1.0));
        vector<bool> visited(vertices);
        bestCost = -1;

        // Generator liczb losowych - przeniesiony na zewnątrz pętli dla wydajności
        random_device rd;
        mt19937 gen(rd());

        // Główna pętla iteracyjna algorytmu
        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            vector<vector<int>> allPaths(numAnts);
            vector<int> allCosts(numAnts, -1);
            int iterationBestCost = -1;  // Dodane: śledzenie najlepszego kosztu w iteracji

            // Iteracja po mrówkach
            for (int ant = 0; ant < numAnts; ++ant) {
                fill(visited.begin(), visited.end(), false);
                vector<int> currentPath;
                currentPath.reserve(vertices + 1);  // Dodane: rezerwacja pamięci
                int currentCost = 0;

                // Wybór losowego miasta początkowego
                uniform_int_distribution<> distrib(0, vertices - 1);
                int startCity = distrib(gen);

                currentPath.push_back(startCity);
                visited[startCity] = true;

                // Budowanie ścieżki
                for (int step = 1; step < vertices; ++step) {
                    int currentCity = currentPath.back();
                    vector<double> probabilities(vertices, 0.0);
                    double sumProbabilities = 0.0;

                    // Obliczanie prawdopodobieństw
                    for (int nextCity = 0; nextCity < vertices; ++nextCity) {
                        if (!visited[nextCity]) {
                            int edgeIndex;
                            if (findEdge(currentCity, nextCity, edgeIndex)) {
                                double pheromone = pheromones[currentCity][nextCity];
                                double distance = *(matrix + currentCity * edges + edgeIndex);
                                double heuristic = 1.0 / distance;
                                probabilities[nextCity] = pow(pheromone, alpha) * pow(heuristic, beta);
                                sumProbabilities += probabilities[nextCity];
                            }
                        }
                    }

                    if (sumProbabilities <= 0.0) {
                        currentCost = -1;
                        break;
                    }

                    // Wybór kolejnego miasta
                    uniform_real_distribution<> random(0.0, sumProbabilities);
                    double randomValue = random(gen);
                    double cumulativeProbability = 0.0;
                    int chosenCity = -1;

                    // Zoptymalizowany wybór miasta
                    for (int nextCity = 0; nextCity < vertices && chosenCity == -1; ++nextCity) {
                        if (!visited[nextCity] && probabilities[nextCity] > 0) {
                            cumulativeProbability += probabilities[nextCity];
                            if (randomValue <= cumulativeProbability) {
                                chosenCity = nextCity;
                            }
                        }
                    }

                    if (chosenCity == -1) {
                        currentCost = -1;
                        break;
                    }

                    int edgeIndex;
                    if (findEdge(currentCity, chosenCity, edgeIndex)) {
                        currentCost += *(matrix + currentCity * edges + edgeIndex);
                        currentPath.push_back(chosenCity);
                        visited[chosenCity] = true;
                    }
                }

                // Powrót do miasta startowego
                if (currentCost != -1) {
                    int edgeIndex;
                    if (findEdge(currentPath.back(), startCity, edgeIndex)) {
                        currentCost += *(matrix + currentPath.back() * edges + edgeIndex);
                        currentPath.push_back(startCity);
                    }
                    else {
                        currentCost = -1;
                    }
                }

                allPaths[ant] = currentPath;
                allCosts[ant] = currentCost;

                // Aktualizacja najlepszego kosztu w iteracji
                if (currentCost != -1 && (iterationBestCost == -1 || currentCost < iterationBestCost)) {
                    iterationBestCost = currentCost;
                }
            }

            // Aktualizacja najlepszego rozwiązania
            for (int ant = 0; ant < numAnts; ++ant) {
                if (allCosts[ant] != -1 && (bestCost == -1 || allCosts[ant] < bestCost)) {
                    bestPath = allPaths[ant];
                    bestCost = allCosts[ant];
                }
            }

            // Aktualizacja feromonów z małą modyfikacją dla lepszej zbieżności
            if (strategy == "CAS") {
                for (int i = 0; i < vertices; ++i) {
                    for (int j = 0; j < vertices; ++j) {
                        pheromones[i][j] *= (1.0 - evaporationRate);
                    }
                }

                // Dodanie feromonów z większym naciskiem na najlepsze ścieżki
                for (int ant = 0; ant < numAnts; ++ant) {
                    if (allCosts[ant] != -1) {
                        double contribution = (allCosts[ant] == iterationBestCost) ?
                            1.5 / allCosts[ant] : 1.0 / allCosts[ant];
                        for (size_t k = 0; k < allPaths[ant].size() - 1; ++k) {
                            int from = allPaths[ant][k];
                            int to = allPaths[ant][k + 1];
                            pheromones[from][to] += contribution;
                        }
                    }
                }
            }
            else if (strategy == "DAS") {
                for (int ant = 0; ant < numAnts; ++ant) {
                    if (allCosts[ant] != -1) {
                        double contribution = 1.0 / allCosts[ant];
                        for (size_t k = 0; k < allPaths[ant].size() - 1; ++k) {
                            int from = allPaths[ant][k];
                            int to = allPaths[ant][k + 1];
                            pheromones[from][to] *= (1.0 - evaporationRate);
                            pheromones[from][to] += contribution;
                        }
                    }
                }
            }
        }
    }
    // Czyszczenie tablicy wierzchołków
    void clearVisited() {
        for (int i = 0; i < vertices; ++i) {
            visited[i] = false;
        }
    }
    // Poszukuje w macierzy odpowiedniej krawędzi pomiędzy dwoma wierzchołkami
    bool findEdge(int vertex1, int vertex2, int& edgeIndex) {
        for (int i = 0; i < edges; i++) {
            if (*(matrix + vertex1 * edges + i) > 0 && *(matrix + vertex2 * edges + i) != 0) {
                edgeIndex = i;
                return true;  // Zwraca indeks krawędzi
            }
        }
        return false;  // Zwraca -1, jeśli nie znaleziono krawędzi
    }
    // Pobieranie wyniku optymalnego z TSP
    int extractOptimalCost(const string& filePath) {
        ifstream file(filePath);
        if (!file.is_open()) {
            throw runtime_error("Could not open the file.");
        }

        string line;
        while (getline(file, line)) {
            // Szukamy linii zawierającej "COMMENT :"
            if (line.find("COMMENT :") != string::npos) {
                // Wyciągamy koszt z komentarza
                size_t startPos = line.find_last_of('(') + 1;
                size_t endPos = line.find_last_of(')');
                if (startPos != string::npos && endPos != string::npos && startPos < endPos) {
                    string costStr = line.substr(startPos, endPos - startPos);
                    return stoi(costStr); // Konwertujemy koszt na liczbę całkowitą
                }
            }
        }

        throw std::runtime_error("Could not find the optimal cost in the file.");
    }
    // Czytanie pliku konfiguracyjnego    
    void readConfigFile(const string& filename, string& graphgeneration, string& dataFile, string& symmetry, string& algorithm, string& optFile, int& ants, double& alpha, double& beta, double& evaporationRate, string& strategy) {
        ifstream file(filename);

        if (!file.is_open()) {
            cerr << "Nie można otworzyć pliku konfiguracyjnego" << endl;
            return;
        }

        string line;
        while (getline(file, line)) {
            // Usuwamy komentarze (tekst po #) oraz białe znaki na początku i końcu
            size_t commentPos = line.find('#');
            if (commentPos != string::npos) {
                line = line.substr(0, commentPos);
            }
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);

            if (line.empty()) {
                continue; // Pomijamy puste linie
            }

            // Parsowanie klucz-wartość
            size_t separatorPos = line.find('=');
            if (separatorPos == string::npos) {
                cerr << "Nieprawidłowy format w linii: " << line << endl;
                continue;
            }

            string key = line.substr(0, separatorPos);
            string value = line.substr(separatorPos + 1);

            // Usuwanie białych znaków wokół klucza i wartości
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            // Przypisanie wartości
            if (key == "graphgeneration") {
                graphgeneration = value;
            }
            else if (key == "dataFile") {
                dataFile = value;
            }
            else if (key == "symmetry") {
                symmetry = value;
            }
            else if (key == "density") {
                density = stoi(value);
            }
            else if (key == "vertices") {
                vertices = stoi(value);
            }
            else if (key == "ants") {
                ants = stoi(value);
            }
            else if (key == "alpha") {
                alpha = stod(value);
            }
            else if (key == "beta") {
                beta = stod(value);
            }
            else if (key == "evaporationRate") {
                evaporationRate = stod(value);
            }
            else if (key == "strategy") {
                strategy = value;
            }
            else if (key == "optFile") {
                optFile = value; // Klucz do obsługi pliku z optymalnym kosztem
            }
            else {
                cerr << "Nieznany klucz w pliku konfiguracyjnym: " << key << endl;
            }
        }
    }
    // Funkcja zapisuje wyniki do pliku csv
    void saveToCSV(const string& filename, const vector<int>& path, int totalCost, double time) {
        std::ofstream file;

        // Otwiera plik w trybie dopisywania (append)
        file.open(filename, ios::app);

        if (!file.is_open()) {
            cerr << "Nie mozna otworzyc pliku: " << filename << endl;
            return;
        }

        // Zapisanie ścieżki do pliku CSV 
        file << "Sciezka:";
        file << endl;
        for (size_t i = 0; i < path.size(); i++) {
            file << path[i];
            if (i != path.size() - 1) {
                file << ",";  // Dodaje przecinek po każdym elemencie oprócz ostatniego
            }
        }

        file << endl;
        file << "Koszt calkowity:";
        file << endl;
        file << totalCost << endl;
        file << "Czas wykonania:";
        file << endl;
        file << time << endl;
        file << endl;

        file.close();

        cout << "Wynik zostal dopisany do pliku: " << filename << endl;
    }

};
int main() {
    /*Plik konfiguracyjny zawiera parametry:
    - losowo/plik
    - symetryczny/asymeryczny
    - bruteforce/nearestneighbour/random
    - 'density'/'plik.txt'
    - 'vertices'/' '*/

    IncidencyMatrix matrixGraph(0, 0, 0);
    double suma = 0;
    chrono::high_resolution_clock::time_point start, end;
    chrono::duration<double, std::milli> elapsed_seconds;
    string filename, dataFile, symmetry, algorithm, graphgeneration, optfile, strategy; // zmienne oreślające nazwę pliku konfiguracyjnego, pliku z danymi, rodzaju grafu oraz algorytmu tsp 
    int iterationcount = 1;

    vector<int> path, bestPath, bestPathManyEdges; //zmienne okreśaljące aktualną ścieżkę, najlepszą ścieżke, oraz najlepszą ścieżke dla startu z każdego wierzchołka 
    vector<int> notToVisit;
    int totalCost = 0;
    bool foundCycle = false;
    int startVertex = 0;
    //int exactTargetCost = 90;
    int bestCost = -1, bestCostManyEdges = -1;
    int count; //zmienna uzywana do okreslenia wierzchołka w algorytmie DFS

    // Parametry algorytmu mrówkowego
    int numAnts;                 // Liczba mrówek
    int maxIterations = 100;     // Maksymalna liczba iteracji
    double alpha, beta, evaporationRate;

    // Startujemy z losowego wierzchołka
    random_device rd;
    mt19937 gen(rd());

    filename = "plik_konfiguracyjny_4.txt";
    matrixGraph.readConfigFile(filename, graphgeneration, dataFile, symmetry, algorithm, optfile, numAnts, alpha, beta, evaporationRate, strategy);

    if (symmetry == "symetryczny") {
        if (graphgeneration == "losowo") {
            matrixGraph.generateNotDirectedArray();
            matrixGraph.fillNotDirectedArray();
        }
        if (graphgeneration == "plik") {
            matrixGraph.readNotDirectedGraphFromFile(dataFile);
        }
        if (graphgeneration == "tsplib") {
            matrixGraph.fillFromTSPLIB(dataFile);
        }
    }
    if (symmetry == "asymetryczny") {
        if (graphgeneration == "losowo") {
            matrixGraph.generateDirectedArray();
            matrixGraph.fillDirectedArray();
        }
        if (graphgeneration == "plik") {
            matrixGraph.readDirectedGraphFromFile(dataFile);
        }
    }
    /*string matrixLocation = "C:\\Users\\wikto\\Downloads\\ALL_tsp\\gr48.tsp\\gr48.tsp";
    matrixGraph.fillFromTSPLIB(matrixLocation);*/
    matrixGraph.isVisited();

    start = chrono::high_resolution_clock::now();

    matrixGraph.antColonyOptimization(bestPath, bestCost, numAnts, maxIterations, alpha, beta, evaporationRate, strategy);

    end = chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    suma = elapsed_seconds.count();
    cout << "Czas wykonania (Tabu Search): " << suma / iterationcount << " millisekund\n" << endl;

    if (bestCost == 0) {
        cout << "Brak cyklu Hamiltona w grafie (Tabu Search)\n";
    }
    else {
        float wynik_optymalny = matrixGraph.extractOptimalCost(optfile);
        cout << "Roznica miedzy wynikiem, a rozwiazaniem optymalnym: " << (float(bestCost) - wynik_optymalny) / wynik_optymalny * 100 << " %\n";
        cout << "Rozwiazanie optymalne: " << wynik_optymalny << "\n";
        cout << "Oto minimalny koszt cyklu: " << bestCost << "\n";
        cout << "Najkrotszy znaleziony cykl: ";
        for (int i : bestPath) {
            cout << i << " ";
        }
        cout << endl;
    }

    // Zapis wyników do pliku CSV
    matrixGraph.saveToCSV("wyniki.csv", bestPath, bestCost, suma / iterationcount);
    if (matrixGraph.edges < 12) {
        matrixGraph.printArray();
    }
    matrixGraph.deleteIncidencyMatrix();
    return 0;

}
