classdef NgILW_SHADE < ALGORITHM
    % <single> <real/integer> <large/none> <constrained/none>
    % Individual-level window method
    % wl_min  --- 10 --- Min Windows length
    methods
        function main(Algorithm,Problem)
            time_start=cputime;
            %% Generate random population
            Population = Problem.Initialization();
            Archive    = [];
            MCR = zeros(Problem.N,1) + 0.5;
            MF  = zeros(Problem.N,1) + 0.5;
            k   = 1;

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
            wl_min = Algorithm.ParameterSet();
            ngilw=NgILW(Problem.D,wl_min,Problem.N,Problem.D,M,C,Population);

            %% recoder
            gen=1;
            Algorithm.metric.conv(gen,1)=Population.best.objs;
            Algorithm.metric.bextX=Population.best.decs;

            %% 备份 F和CR
            CR  = randn(Problem.N,1).*sqrt(0.1) + MCR(randi(end,Problem.N,1));
            CR  = repmat(max(0,min(1,CR)),1,Problem.D);
            F   = min(1,trnd(1,Problem.N,1).*sqrt(0.1) + MF(randi(end,Problem.N,1)));
            while any(F<=0)
                F(F<=0) = min(1,trnd(1,sum(F<=0),1).*sqrt(0.1) + MF(randi(end,sum(F<=0),1)));
            end
            F = repmat(F,1,Problem.D);
            CR_previous=CR;
            F_previous=F;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Generate offspring
                [~,rank] = sort(FitnessSingle(Population));
                Xpb = Population(rank(ceil(rand(1,Problem.N).*max(2,rand(1,Problem.N)*0.2*Problem.N)))).decs;
                Xr1 = Population(randi(end,1,Problem.N)).decs;
                P   = [Population,Archive];
                Xr2 = P(randi(end,1,Problem.N)).decs;
                CR  = randn(Problem.N,1).*sqrt(0.1) + MCR(randi(end,Problem.N,1));
                CR  = repmat(max(0,min(1,CR)),1,Problem.D);
                F   = min(1,trnd(1,Problem.N,1).*sqrt(0.1) + MF(randi(end,Problem.N,1)));
                while any(F<=0)
                    F(F<=0) = min(1,trnd(1,sum(F<=0),1).*sqrt(0.1) + MF(randi(end,sum(F<=0),1)));
                end
                F = repmat(F,1,Problem.D);

                Site         = rand(size(CR)) < CR;
                OffDec       = Population.decs;
                OffDec(Site) = OffDec(Site) + F(Site).*(Xpb(Site)-OffDec(Site)+Xr1(Site)-Xr2(Site));
                
                %% 当前维度操作
                offDec_window=Population.decs;
                for i =1:Problem.N
                    window=ngilw.windows{i};
                    offDec_window(i,window)=OffDec(i,window);
                    CR_previous(i,window)=CR(i,window);
                    F_previous(i,window)=F(i,window);
                end

                Offspring    = Problem.Evaluation(offDec_window);

                % Update the population and archive
                delta   = FitnessSingle(Population) - FitnessSingle(Offspring);
                replace = delta > 0;
                Archive = [Archive,Population(replace)];
                Archive = Archive(randperm(end,min(end,Problem.N)));
                Population(replace) = Offspring(replace);
                % Update the parameters
                if any(replace)
                    w      = delta(replace)./sum(delta(replace));
                    MCR(k) = w'*CR_previous(replace,1);
                    MF(k)  = (w'*F_previous(replace,1).^2)./(w'*F_previous(replace,1));
                    k      = mod(k,length(MCR)) + 1;
                end

                 %% 窗口进化
                ngilw=ngilw.next_step(Population,gen);

                %% recoder
                gen=gen+1;
                Algorithm.metric.conv(gen,1)=Population.best.objs;
                Algorithm.metric.bextX=Population.best.decs;

                %% 输出
                if mod(gen,100)==0 || Problem.FE==Problem.maxFE
                    fprintf('%s  迭代次数: %d  最优值：%.4e  window length: %d  替换个数: %d\n',datestr(now,31),...
                        gen,Population.best.objs,length(ngilw.windows{1}),sum(replace));
                end
            end
            Algorithm.metric.time_total=time_start-cputime;
        end
    end
end