function QLPSO()
    % Map parameters
    mapSize = [20, 20];  % Map size (10x10 grid)
    start = [1, 1];  % Start position
    goal = [20, 20];  % Goal position
    obstacleRatio = 0.2;  % Percentage of obstacles in the map
    
    % Generate random map
    map = GenerateRandomMap(mapSize, start, goal, obstacleRatio);
    
    % Initialize parameters
    numParticles = 40;
    maxIterations = 200;
    learningRate = 0.6;
    discountFactor = 0.8;
    explorationRate = 0.3;
    
    % PSO parameters
    inertiaWeight = 0.7;
    cognitiveConstant = 1.5;
    socialConstant = 0.6;
    
    % Q-Learning parameters
    QTable = zeros(numParticles, 4);  % Example with 4 actions
    state = randi([1, numParticles], numParticles, 1);  % Initial random states
    
    % Particle initialization
    particles = repmat(start, numParticles, 1);
    velocities = zeros(numParticles, 2);
    personalBestPositions = particles;
    globalBestPosition = start;
    personalBestScores = inf(numParticles, 1);
    globalBestScore = inf;
    
    % Create figure for visualization
    fig = figure('Name', 'Q-Learning PSO Path Planning', 'NumberTitle', 'off');
    
    % Start timer
    tic;
    
    % Main loop
    for iter = 1:maxIterations
        for i = 1:numParticles
            % Evaluate particle fitness
            fitness = EvaluateFitness(particles(i, :), goal, map);
            
            % Update personal and global best
            if fitness < personalBestScores(i)
                personalBestScores(i) = fitness;
                personalBestPositions(i, :) = particles(i, :);
            end
            if fitness < globalBestScore
                globalBestScore = fitness;
                globalBestPosition = particles(i, :);
            end
            
            % Q-Learning action selection
            if rand < explorationRate
                action = randi(4);  % Random action (exploration)
            else
                [~, action] = max(QTable(state(i), :));  % Best action (exploitation)
            end
            
            % Update velocities and positions based on action
            switch action
                case 1
                    velocities(i, :) = [1, 0];  % Move right
                case 2
                    velocities(i, :) = [0, 1];  % Move up
                case 3
                    velocities(i, :) = [-1, 0];  % Move left
                case 4
                    velocities(i, :) = [0, -1];  % Move down
            end
            
            newPos = particles(i, :) + velocities(i, :);
            
            % Check if the new position is valid
            if IsValidPosition(newPos, map)
                particles(i, :) = newPos;
            end
            
            % Calculate reward and update Q-Table
            newFitness = EvaluateFitness(particles(i, :), goal, map);
            reward = -newFitness;
            [~, bestNextAction] = max(QTable(state(i), :));
            QTable(state(i), action) = QTable(state(i), action) ...
                + learningRate * (reward + discountFactor * QTable(state(i), bestNextAction) - QTable(state(i), action));
        end
        
        % Display iteration information
        fprintf('Iteration %d: Best Fitness = %.4f\n', iter, globalBestScore);
        
        % Visualize current state
        VisualizeMap(fig, map, particles, globalBestPosition, start, goal, iter, maxIterations);
        drawnow;

        % Check if any particle has reached the goal
        if any(all(particles == goal, 2))
            break;
        end
    end
    
    % Stop timer and calculate elapsed time
    elapsedTime = toc;
    
    % Final result
    fprintf('\nOptimization completed.\n');
    fprintf('Global Best Position: [%d, %d]\n', globalBestPosition(1), globalBestPosition(2));
    fprintf('Global Best Fitness: %.4f\n', globalBestScore);
    fprintf('Total time: %.2f seconds\n', elapsedTime);
    
    % Visualize final result
    VisualizeMap(fig, map, particles, globalBestPosition, start, goal, maxIterations, maxIterations, true);
end

function map = GenerateRandomMap(mapSize, start, goal, obstacleRatio)
    map = zeros(mapSize);
    numObstacles = floor(prod(mapSize) * obstacleRatio);
    obstacles = randperm(prod(mapSize), numObstacles);
    map(obstacles) = 1;  % Set obstacles
    map(start(1), start(2)) = 0;  % Ensure start is free
    map(goal(1), goal(2)) = 0;  % Ensure goal is free
end

function isValid = IsValidPosition(position, map)
    if position(1) < 1 || position(1) > size(map, 1) || position(2) < 1 || position(2) > size(map, 2)
        isValid = false;
    else
        isValid = map(position(1), position(2)) == 0;
    end
end

function fitness = EvaluateFitness(position, goal, map)
    if position == goal
        fitness = 0;
    else
        fitness = norm(position - goal);
        if map(position(1), position(2)) == 1
            fitness = fitness + 100;  % Penalize hitting obstacles
        end
    end
end

function VisualizeMap(fig, map, particles, globalBestPosition, start, goal, currentIter, maxIterations, isFinal)
    if nargin < 9
        isFinal = false;
    end
    
    figure(fig);
    clf;
    
    % Plot the map
    imagesc(map');
    colormap([1 1 1; 0.7 0.7 0.7]);  % White for free space, gray for obstacles
    hold on;
    
    % Plot particles
    scatter(particles(:, 1), particles(:, 2), 30, 'bo', 'filled', 'MarkerEdgeColor', 'k');
    
    % Plot global best position
    scatter(globalBestPosition(1), globalBestPosition(2), 100, 'ro', 'filled', 'MarkerEdgeColor', 'k');
    
    % Plot start and goal
    scatter(start(1), start(2), 100, 'gs', 'filled', 'MarkerEdgeColor', 'k');
    scatter(goal(1), goal(2), 100, 'ys', 'filled', 'MarkerEdgeColor', 'k');
    
    % Add labels and title
    xlabel('X');
    ylabel('Y');
    if isFinal
        title('Final Path', 'FontSize', 14, 'FontWeight', 'bold');
    else
        title(sprintf('Q-Learning PSO Path Planning (Iteration %d/%d)', currentIter, maxIterations), 'FontSize', 14, 'FontWeight', 'bold');
    end
    
    % Add legend
    legend('Particles', 'Best Position', 'Start', 'Goal', 'Location', 'northeastoutside');
    
    % Adjust axis
    axis equal tight;
    set(gca, 'YDir', 'normal');
    
    % Add grid
    grid on;
    
    % Improve overall appearance
    set(gca, 'FontSize', 12);
    set(gcf, 'Color', 'w');
    
    % Add colorbar for obstacle representation
    colorbar('Ticks', [0.25, 0.85], 'TickLabels', {'Free', 'Obstacle'});
    
    hold off;
end