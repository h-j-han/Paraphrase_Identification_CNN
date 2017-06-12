
function createModel(mdl, vocsize, Dsize, nout, KKw)
    	print(mdl)
        print(vocsize) --10000
        print(Dsize)   --300
        print(nout)    --6
        print(KKw)     --3
        -- define model to train
    	local network = nn.Sequential()
    	local featext = nn.Sequential()
    	local classifier = nn.Sequential()

    	local conCon1 = nn.Sequential()
    	local conCon2 = nn.Sequential()
    	local conCon3 = nn.Sequential()
    	local conCon4 = nn.Sequential()

    	local parallelConcat1 = nn.Concat(1)
    	local parallelConcat2 = nn.Concat(1)
    	local parallelConcat3 = nn.Concat(1)
    	local parallelConcat4 = nn.Concat(1)
    	local parallelConcat5 = nn.Concat(1)

    	local D     = Dsize --300 
    	local kW    = KKw   --3
    	local dW    = 1      
    	local noExtra = false
    	local nhid1 = 250 
    	local nhid2 = 250 
    	local NumFilter = D --300
    	local pR = 2 --if 1 relu if 2 tanh
    	local layers=1
	    
    if mdl == 'deepQueryRankingNgramSimilarityOnevsGroupMaxMinMeanLinearExDGpPoinPercpt' then
		dofile "PaddingReshape.lua"
		
		deepQuery=nn.Sequential()
   		D = Dsize --300
		----------------------------------------------------------------------  
        -- Modeling sentence Feature extraction
		----------------------------------------------------------------------  
		-- Holistic Conv
        -- MAX --
        local incep1max = nn.Sequential()
		incep1max:add(nn.TemporalConvolution(D,NumFilter,1,dw)) --ws = 1
		if pR == 1 then
			incep1max:add(nn.PReLU())
		else 
		  	incep1max:add(nn.Tanh())
		end		  
		incep1max:add(nn.Max(1))
		incep1max:add(nn.Reshape(NumFilter,1))
        --??
		local incep2max = nn.Sequential()
		incep2max:add(nn.Max(1))
		incep2max:add(nn.Reshape(NumFilter,1))			  
		local combineDepth = nn.Concat(2)
		combineDepth:add(incep1max)
		combineDepth:add(incep2max)
		  
		local ngram = kW --3                
		for cc = 2, ngram do
		    local incepMax = nn.Sequential()
		    if not noExtra then --false so yes do if
		    	incepMax:add(nn.TemporalConvolution(D,D,1,dw)) --set
                --print('incepMax:add(nn.TemporalConvolution(D,D,1,dw))')
		    	if pR == 1 then
				    incepMax:add(nn.PReLU())
			    else 
				    incepMax:add(nn.Tanh())
                    --print("incepMax:add(nn.Tanh())")
			    end
		    end
		    incepMax:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
		    if pR == 1 then
			  	incepMax:add(nn.PReLU())
			else 
			  	incepMax:add(nn.Tanh())
			end 
		    incepMax:add(nn.Max(1))
		    incepMax:add(nn.Reshape(NumFilter,1))		    		    
		    
            combineDepth:add(incepMax)		    
		end  		  
		
        -- MIN --
		local incep1min = nn.Sequential()
		incep1min:add(nn.TemporalConvolution(D,NumFilter,1,dw))
		if pR == 1 then
			incep1min:add(nn.PReLU())
		else 
			incep1min:add(nn.Tanh())
		end		  
		incep1min:add(nn.Min(1))
		incep1min:add(nn.Reshape(NumFilter,1))		  
		local incep2min = nn.Sequential()
		incep2min:add(nn.Min(1))
		incep2min:add(nn.Reshape(NumFilter,1))		  
		combineDepth:add(incep1min)
		combineDepth:add(incep2min)
		  
		for cc = 2, ngram do		    
			local incepMin = nn.Sequential()
            if not noExtra then
            incepMin:add(nn.TemporalConvolution(D,D,1,dw)) --set
                if pR == 1 then
                    incepMin:add(nn.PReLU())
                else 
                    incepMin:add(nn.Tanh())
                end
            end		  
            incepMin:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
            if pR == 1 then
                incepMin:add(nn.PReLU())
            else 
                incepMin:add(nn.Tanh())
            end
            incepMin:add(nn.Min(1))
            incepMin:add(nn.Reshape(NumFilter,1))		    		  	
		  	
            combineDepth:add(incepMin)		      		    
		end  
		
        -- MEAN -- 
		local incep1mean = nn.Sequential()
		incep1mean:add(nn.TemporalConvolution(D,NumFilter,1,dw))
		if pR == 1 then
			incep1mean:add(nn.PReLU())
		else 
			incep1mean:add(nn.Tanh())
		end
		incep1mean:add(nn.Mean(1))
		incep1mean:add(nn.Reshape(NumFilter,1))		    		  		  
		local incep2mean = nn.Sequential()
		incep2mean:add(nn.Mean(1))
		incep2mean:add(nn.Reshape(NumFilter,1))		  
		combineDepth:add(incep1mean)
		combineDepth:add(incep2mean)		  
		for cc = 2, ngram do
		    local incepMean = nn.Sequential()
		    if not noExtra then
		    	incepMean:add(nn.TemporalConvolution(D,D,1,dw)) --set
		    	if pR == 1 then
				    incepMean:add(nn.PReLU())
			    else 
				    incepMean:add(nn.Tanh())
			    end
		    end
		    incepMean:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
		    if pR == 1 then
		    	incepMean:add(nn.PReLU())
		    else 
			incepMean:add(nn.Tanh())
		    end
		    incepMean:add(nn.Mean(1))
		    incepMean:add(nn.Reshape(NumFilter,1))			    
		    
            combineDepth:add(incepMean)	
		end  
        
		----------------------------------------------------------------------  
		-- PER Dimension Conv
        -- MAX
        local conceptFNum = 20
		for cc = 1, ngram do
			local perConcept = nn.Sequential()
			perConcept:add(nn.PaddingReshape(2,2)) --set
		    perConcept:add(nn.SpatialConvolutionMM(1,conceptFNum,1,cc)) --set
		    perConcept:add(nn.Max(2)) --set
		    if pR == 1 then
			 	perConcept:add(nn.PReLU())
			else 
			 	perConcept:add(nn.Tanh())
			end
			perConcept:add(nn.Transpose({1,2}))
		    
            combineDepth:add(perConcept)	
		end
		-- MIN
		for cc = 1, ngram do
			local perConcept = nn.Sequential()
            perConcept:add(nn.PaddingReshape(2,2)) --set
            perConcept:add(nn.SpatialConvolutionMM(1,conceptFNum,1,cc)) --set
            perConcept:add(nn.Min(2)) --set
            if pR == 1 then
                perConcept:add(nn.PReLU())
            else 
                perConcept:add(nn.Tanh())
            end
			perConcept:add(nn.Transpose({1,2}))
		    
            combineDepth:add(perConcept)	
		end
		  
		
		----------------------------------------------------------------------  
		-- Siamese Net
        ----------------------------------------------------------------------  
        featext:add(combineDepth)		
		local items = (ngram+1)*3 -- 12 		
		local separator = items+2*conceptFNum*ngram -- 12 + 2*20*3 = 120 + 12 = 132
		local sepModel = 0 
		if sepModel == 1 then  
			modelQ= featext:clone()
		else
			modelQ= featext:clone('weight','bias','gradWeight','gradBias')
		end
		paraQuery=nn.ParallelTable()
		    paraQuery:add(modelQ)
          	paraQuery:add(featext)			
        deepQuery:add(paraQuery) 
		deepQuery:add(nn.JoinTable(2)) 
		----------------------------------------------------------------------  
        -- Similarity Measurement
		----------------------------------------------------------------------  
		local coad = 0 -- count add d = count  d:add
		d=nn.Concat(1) 
		----------------------------------------------------------------------  
        -- Algorithm 2 Group A Comparison Unit 1 (output 302)
		for i=1,items do
            -- MAX
  			if i <= items/3 then 					
	  			for j=1,items/3 do
	  				local connection = nn.Sequential()
					local minus=nn.Concat(2)
					local c1=nn.Sequential()
					local c2=nn.Sequential()
					c1:add(nn.Select(2,i)) -- == D, not D*1
					c1:add(nn.Reshape(NumFilter,1)) --D*1 here					
					c2:add(nn.Select(2,separator+j))					
					c2:add(nn.Reshape(NumFilter,1))
					minus:add(c1)
					minus:add(c2)
					connection:add(minus) -- D*2						
					local similarityC=nn.Concat(1) -- multi similarity criteria			
					local s1=nn.Sequential()
					s1:add(nn.SplitTable(2))
					s1:add(nn.PairwiseDistance(2)) -- scalar
					local s2=nn.Sequential()
					if 1 < 3 then
						s2:add(nn.SplitTable(2))
					else
						s2:add(nn.Transpose({1,2})) 
						s2:add(nn.SoftMax())
						s2:add(nn.SplitTable(1))										
					end						
					s2:add(nn.CsDis()) -- scalar
					local s3=nn.Sequential()
					s3:add(nn.SplitTable(2))
					s3:add(nn.CSubTable()) -- linear
					s3:add(nn.Abs()) -- linear						
					similarityC:add(s1)
					similarityC:add(s2)					
					similarityC:add(s3)
					connection:add(similarityC) -- scalar											
					d:add(connection)
		            coad = coad + 1
				end
            -- MIN
            elseif i <= 2*items/3 then				
                for j=1+items/3, 2*items/3 do
                    local connection = nn.Sequential()
                    local minus=nn.Concat(2)
                    local c1=nn.Sequential()
                    local c2=nn.Sequential()
                    c1:add(nn.Select(2,i)) -- == NumFilter, not NumFilter*1
                    c1:add(nn.Reshape(NumFilter,1)) --NumFilter*1 here
                    c2:add(nn.Select(2,separator+j))
                    c2:add(nn.Reshape(NumFilter,1))
                    minus:add(c1)
                    minus:add(c2)
                    connection:add(minus) -- D*2						
                    local similarityC=nn.Concat(1) -- multi similarity criteria			
                    local s1=nn.Sequential()
                    s1:add(nn.SplitTable(2))
                    s1:add(nn.PairwiseDistance(2)) -- scalar
                    local s2=nn.Sequential()			
                    if 1 < 3 then
                        s2:add(nn.SplitTable(2))
                    else
                        s2:add(nn.Transpose({1,2})) -- D*2 -> 2*D
                        s2:add(nn.SoftMax())
                        s2:add(nn.SplitTable(1))										
                    end									
                    s2:add(nn.CsDis()) -- scalar						
                    local s3=nn.Sequential()
                    s3:add(nn.SplitTable(2))
                    s3:add(nn.CSubTable()) -- linear
                    s3:add(nn.Abs()) -- linear						
                    similarityC:add(s1)
                    similarityC:add(s2)					
                    similarityC:add(s3)
                    connection:add(similarityC) -- scalar												
                    d:add(connection)						
		            coad = coad + 1
                end
            -- MEAN
            else 
                for j=1+2*items/3, items do
                    local connection = nn.Sequential()
                    local minus=nn.Concat(2)
                    local c1=nn.Sequential()
                    local c2=nn.Sequential()
                    c1:add(nn.Select(2,i)) -- == D, not D*1
                    c1:add(nn.Reshape(NumFilter,1)) --D*1 here
                    c2:add(nn.Select(2,separator+j))
                    c2:add(nn.Reshape(NumFilter,1))
                    minus:add(c1)
                    minus:add(c2)
                    connection:add(minus) -- D*2						
                    local similarityC=nn.Concat(1) -- multi similarity criteria			
                    local s1=nn.Sequential()
                    s1:add(nn.SplitTable(2))
                    s1:add(nn.PairwiseDistance(2)) -- scalar
                    local s2=nn.Sequential()					
                    if 1 < 3 then
                        s2:add(nn.SplitTable(2))
                    else
                        s2:add(nn.Transpose({1,2})) -- D*2 -> 2*D
                        s2:add(nn.SoftMax())
                        s2:add(nn.SplitTable(1))										
                    end							
                    s2:add(nn.CsDis()) -- scalar
                    local s3=nn.Sequential()
                    s3:add(nn.SplitTable(2))
                    s3:add(nn.CSubTable()) -- linear
                    s3:add(nn.Abs()) -- linear						
                    similarityC:add(s1)
                    similarityC:add(s2)					
                    similarityC:add(s3)					
                    connection:add(similarityC) -- scalar											
                    d:add(connection)						
		            coad = coad + 1
                end		
            end
		end
        --print('d:added ' .. coad) 48 = 12 * 4
		----------------------------------------------------------------------  
        -- Algorithm 1 Group A Comparison Unit 2 (output 2)
				  				
        for i=1,NumFilter do
            for j=1,3 do 
                local connection = nn.Sequential()
                connection:add(nn.Select(1,i)) -- == 2items
                connection:add(nn.Reshape(2*separator,1)) --2items*1 here					
                local minus=nn.Concat(2)
                local c1=nn.Sequential()
                local c2=nn.Sequential()
                if j == 1 then 
                    c1:add(nn.Narrow(1,1,ngram+1)) -- first half (items/3)*1
                    c2:add(nn.Narrow(1,separator+1,ngram+1)) -- first half (items/3)*1
                elseif j == 2 then
                    c1:add(nn.Narrow(1,ngram+2,ngram+1)) -- 
                    c2:add(nn.Narrow(1,separator+ngram+2,ngram+1)) 
                else
                    c1:add(nn.Narrow(1,2*(ngram+1)+1,ngram+1)) 
                    c2:add(nn.Narrow(1,separator+2*(ngram+1)+1,ngram+1)) --each is ngram+1 portion (max or min or mean)
                end						
                
                minus:add(c1)
                minus:add(c2)
                connection:add(minus) -- (items/3)*2					
                local similarityC=nn.Concat(1) 	
                local s1=nn.Sequential()
                s1:add(nn.SplitTable(2))
                s1:add(nn.PairwiseDistance(2)) -- scalar
                local s2=nn.Sequential()					
                if 1 >= 2 then
                    s2:add(nn.Transpose({1,2})) -- (items/3)*2 -> 2*(items/3)
                    s2:add(nn.SoftMax()) --for softmax have to do transpose from (item/3)*2 -> 2*(item/3)
                    s2:add(nn.SplitTable(1)) --softmax only works on row
                else
                    s2:add(nn.SplitTable(2)) --(items/3)*2
                end
                s2:add(nn.CsDis()) -- scalar
                similarityC:add(s1)
                similarityC:add(s2)					
                connection:add(similarityC) -- scalar											
                d:add(connection)				
		        coad = coad + 1
            end
        end			
        -- print('d:added ' .. coad) 900=300*3

		----------------------------------------------------------------------  
        -- Algorithm 2 Group B Comparison Unit 1 (output 302)

        for i=items+1,separator do --120 times 13~132 : Group B output
            local connection = nn.Sequential()
            local minus=nn.Concat(2)
            local c1=nn.Sequential()
            local c2=nn.Sequential()
            c1:add(nn.Select(2,i)) -- == D, not D*1
            c1:add(nn.Reshape(NumFilter,1)) --D*1 here
            c2:add(nn.Select(2,separator+i))
            c2:add(nn.Reshape(NumFilter,1))
            minus:add(c1)
            minus:add(c2)
            connection:add(minus) -- D*2						
            local similarityC=nn.Concat(1) 			
            local s1=nn.Sequential()
            s1:add(nn.SplitTable(2))
            s1:add(nn.PairwiseDistance(2)) -- scalar
            local s2=nn.Sequential()					
            if 1 < 3 then
                s2:add(nn.SplitTable(2))
            else
                s2:add(nn.Transpose({1,2})) 
                s2:add(nn.SoftMax())
                s2:add(nn.SplitTable(1))										
            end							
            s2:add(nn.CsDis()) -- scalar
            local s3=nn.Sequential()
            s3:add(nn.SplitTable(2))
            s3:add(nn.CSubTable()) -- linear
            s3:add(nn.Abs()) -- linear						
            similarityC:add(s1)
            similarityC:add(s2)					
            similarityC:add(s3)					
            connection:add(similarityC) -- scalar											
            d:add(connection)		
		    coad = coad + 1
        end
	  	
        --print('d:added ' .. coad)
		----------------------------------------------------------------------  
        
        deepQuery:add(d)	    
        return deepQuery	
	end
end

