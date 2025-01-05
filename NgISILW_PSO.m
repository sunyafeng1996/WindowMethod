classdef NgISILW_PSO < ALGORITHM
    % <single> <real/integer> <large/none> <constrained/none>
    % Particle swarm optimization
    % W --- 0.4 --- Inertia weight

    %------------------------------- Reference --------------------------------
    % R. Eberhart and J. Kennedy, A new optimizer using particle swarm theory,
    % Proceedings of the International Symposium on Micro Machine and Human
    % Science, 1995, 39-43.
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
            W = Algorithm.ParameterSet(0.4);

            %% Generate random population
            Population = Problem.Initialization();
            Pbest      = Population;
            [~,best]   = min(FitnessSingle(Pbest));
            Gbest      = Pbest(best);

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
            Algorithm.metric.conv(gen,1)=Gbest.objs;
            Algorithm.metric.bextX=Gbest.decs;
            Algorithm.metric.Min_value=Gbest.objs;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Population     = OperatorPSO(Problem,Population,Pbest,Gbest,W);

                ParticleDec = Population.decs;
                PbestDec    = Pbest.decs;
                GbestDec    = Gbest.decs;
                [N,D]       = size(ParticleDec);
                ParticleVel = Population.adds(zeros(N,D));

                %% Particle swarm optimization
                r1        = rand(N,D);
                r2        = rand(N,D);
                OffVel    = W.*ParticleVel + r1.*(PbestDec-ParticleDec) + r2.*(GbestDec-ParticleDec);
                OffDec    = ParticleDec + OffVel;

                %% operator now dim
                offDec_window=Population.decs;
                for i =1:Problem.N
                    window=ngisilw.windows{i};
                    offDec_window(i,window)=OffDec(i,window);
                end
                Population = Problem.Evaluation(offDec_window);

                replace        = FitnessSingle(Pbest) > FitnessSingle(Population);
                Pbest(replace) = Population(replace);
                [~,best]       = min(FitnessSingle(Pbest));
                Gbest          = Pbest(best);
                
                %% update windows
                ngisilw=ngisilw.next_step(Population,gen);
                
                gen=gen+1;
                Algorithm.metric.conv(gen,1)=Gbest.objs;
                Algorithm.metric.bextX=Gbest.decs;
                Algorithm.metric.Min_value=Gbest.objs;
            end
        end
    end
end