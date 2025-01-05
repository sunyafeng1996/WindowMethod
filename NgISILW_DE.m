classdef NgISILW_DE < ALGORITHM
% <single> <real/integer> <large/none> <constrained/none>
% Differential evolution

%------------------------------- Reference --------------------------------
% R. Storn and K. Price, Differential evolution-a simple and efficient
% heuristic for global optimization over continuous spaces, Journal of
% Global Optimization, 1997, 11(4): 341-359.
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
            % [CR,F] = Algorithm.ParameterSet(0.9,0.5);
            CR=0.9;F=0.5;
            
            %% Generate random population
            Population = Problem.Initialization();

            %% window hyperparameter
            t=40; wl_min=10;
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
            ngisilw=NgISILW(Problem.D,wl_min,Problem.N,Problem.D,M,C,Population,t);

            gen=1;
            Algorithm.metric.conv(gen,1)=Population.best.objs;
            Algorithm.metric.bextX=Population.best.decs;
            Algorithm.metric.Min_value=Population.best.objs;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,2*Problem.N,FitnessSingle(Population));
                OffDec  = OperatorDE(Problem,Population.decs,Population(MatingPool(1:end/2)).decs,Population(MatingPool(end/2+1:end)).decs,{CR,F,0,0});
                
                %% operator now dim
                offDec_window=Population.decs;
                for i =1:Problem.N
                    window=ngisilw.windows{i};
                    offDec_window(i,window)=OffDec(i,window);
                end
                Offspring = Problem.Evaluation(offDec_window);

                replace             = FitnessSingle(Population) > FitnessSingle(Offspring);
                Population(replace) = Offspring(replace);

                %% update windows
                ngisilw=ngisilw.next_step(Population,gen);

                gen=gen+1;
                Algorithm.metric.conv(gen,1)=Population.best.objs;
                Algorithm.metric.bextX=Population.best.decs;
                Algorithm.metric.Min_value=Population.best.objs;
            end
        end
    end
end