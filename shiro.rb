tree = '(Root (Spine (Neck (Head))
                     (RClavicle (RUpperArm (RLowerArm (RHand))))
                     (LClavicle (LUpperArm (LLowerArm (LHand)))))
              (RHip (RUpperLeg (RLowerLeg (RFoot))))
              (LHip (LUpperLeg (LLowerLeg (LFoot)))))'
tree.scan(/ \((\w+)/){puts"#$+ . #{a=$`.split*'';1while a.sub!(/\(\w+\)/,'');a[/\w+$/]}"}
