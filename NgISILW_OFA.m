classdef NgISILW_OFA < ALGORITHM
% <single> <real/integer> <large/none> <constrained/none>
% Optimal foraging algorithm

%------------------------------- Reference --------------------------------
% G. Zhu and W. Zhang, Optimal foraging algorithm for global optimization,
% Applied Soft Computing, 2017, 51: 294-313.
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
            %% Generate random population
            Population = Problem.Initialization();
            [~,rank]   = sort(FitnessSingle(Population));
            Population = Population(rank);
            
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
                PopDec     = Population.decs;
                OffDec     = PopDec + Problem.FE./Problem.maxFE.*(rand(size(PopDec))-rand(size(PopDec))).*(PopDec-PopDec([end,floor(unifrnd(ones(1,end-1),2:end))],:));
                
                %% operator now dim
                offDec_window=Population.decs;
                for i =1:Problem.N
                    window=ngisilw.windows{i};
                    offDec_window(i,window)=OffDec(i,window);
                end
                Offspring = Problem.Evaluation(offDec_window);
                
                lambda     = rand(Problem.N,1);
                replace    = lambda.*FitnessSingle(Offspring)./(1+lambda*ceil(Problem.FE/Problem.N)) < FitnessSingle(Population)/ceil(Problem.FE/Problem.N);
                Population(replace) = Offspring(replace);
                [~,rank]   = sort(FitnessSingle(Population));
                Population = Population(rank);
                
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