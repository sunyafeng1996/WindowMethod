classdef NgISILW_MOEADDE < ALGORITHM
% <multi/many> <real/integer><large>
% MOEA/D based on differential evolution
% delta --- 0.9 --- The probability of choosing parents locally
% nr    ---   2 --- Maximum number of solutions replaced by each offspring

%------------------------------- Reference --------------------------------
% H. Li and Q. Zhang, Multiobjective optimization problems with complicated
% Pareto sets, MOEA/D and NSGA-II, IEEE Transactions on Evolutionary
% Computation, 2009, 13(2): 284-302.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [delta,nr] = Algorithm.ParameterSet(0.9,2);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);

            %% window hyperparameter
            M=size(Population.objs,2);
            if size(Population.cons,2)==1
                if sum(Population.cons)==0
                    C=0;
                else
                    C=1;
                end
            else
                C=size(Population.cons,2);
            end
            wl_min=100;t=40;
            ngisilw=NgISILW(Problem.D,wl_min,Problem.N,Problem.D,M,C,Population,t);

            %% recoder
            Algorithm.metric.generation=1;
            Algorithm.metric.conv_igd(Algorithm.metric.generation,1)=Problem.CalMetric('IGD',Population);
            Algorithm.metric.lastIGD=Algorithm.metric.conv_igd(Algorithm.metric.generation,1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % For each solution
                for i = 1 : Problem.N
                    % Choose the parents
                    if rand < delta
                        P = B(i,randperm(end));
                    else
                        P = randperm(Problem.N);
                    end

                    % Generate an offspring
                    OffDec = OperatorDE(Problem,Population(i).decs,Population(P(1)).decs,Population(P(2)).decs);

                    % operator now dim
                    offDec_window=Population(i).decs;
                    window=ngisilw.windows{i};
                    offDec_window(window)=OffDec(window);
                    Offspring = Problem.Evaluation(offDec_window);

                    % Update the ideal point
                    Z = min(Z,Offspring.obj);

                    % Update the solutions in P by Tchebycheff approach
                    g_old = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                    g_new = max(repmat(abs(Offspring.obj-Z),length(P),1).*W(P,:),[],2);
                    Population(P(find(g_old>=g_new,nr))) = Offspring;
                end
                
                %% update windows
                ngisilw=ngisilw.next_step(Population,Algorithm.metric.generation);
                
                % recoder
                Algorithm.metric.generation=Algorithm.metric.generation+1;
                Algorithm.metric.conv_igd(Algorithm.metric.generation,1)=Problem.CalMetric('IGD',Population);
                Algorithm.metric.lastIGD=Algorithm.metric.conv_igd(Algorithm.metric.generation,1);

                % output
                if mod(Algorithm.metric.generation,100)==0 || Problem.FE==Problem.maxFE
                    fprintf('%s  generator: %d  IGD：%.4e  window length: %d\n',datestr(now,31),...
                        Algorithm.metric.generation,Algorithm.metric.lastIGD,length(ngisilw.windows{1}));
                end
            end
        end
    end
end