from offlinerlkit.policy.base_policy import BasePolicy
import sys
sys.path.append('/lichenghao/lch/transfer_rl-main/OfflineRL-Kit/offlinerlkit')

# model free
from offlinerlkit.policy.model_free.sac import SACPolicy
from offlinerlkit.policy.model_free.td3 import TD3Policy
from offlinerlkit.policy.model_free.cql import CQLPolicy
from offlinerlkit.policy.model_free.iql import IQLPolicy
from offlinerlkit.policy.model_free.mcq import MCQPolicy
from offlinerlkit.policy.model_free.cqldqn import CQLDQNPolicy
from offlinerlkit.policy.model_free.td3bc import TD3BCPolicy
from offlinerlkit.policy.model_free.edac import EDACPolicy

# model based
from offlinerlkit.policy.model_based.mopo import MOPOPolicy
from offlinerlkit.policy.model_based.mopo_tta import MOPOTTAPolicy
from offlinerlkit.policy.model_based.mobile import MOBILEPolicy
from offlinerlkit.policy.model_based.rambo import RAMBOPolicy
from offlinerlkit.policy.model_based.combo import COMBOPolicy

# model free tta
from offlinerlkit.policy.model_free.sac_tta import SACTTAPolicy
from offlinerlkit.policy.model_free.sac_tta_multul import SACTTAMULTULPolicy
from offlinerlkit.policy.model_free.mcq_tta import MCQTTAPolicy
from offlinerlkit.policy.model_free.cql_tta import CQLTTAPolicy
from offlinerlkit.policy.model_free.iql_tta import IQLTTAPolicy
from offlinerlkit.policy.model_free.mcq_tta_buffer import MCQTTABUFFERPolicy
from offlinerlkit.policy.model_free.mcq_tta_multul import MCQTTAMULTULPolicy
from offlinerlkit.policy.model_free.mcq_rdropout import MCQRDropoutPolicy
from offlinerlkit.policy.model_free.sac_discrete import SACPolicyDiscrete
# from offlinerlkit.policy.model_free.cql_discrete import CQLPolicyDiscrete
from offlinerlkit.policy.model_free.cql_discrete_q1 import CQLPolicyDiscreteQ
from offlinerlkit.policy.model_free.cql_discrete_actor import CQLPolicyDiscreteActor
from offlinerlkit.policy.model_free.cql_discrete_actor100 import CQLPolicyDiscreteActor100
from offlinerlkit.policy.model_free.sac_discrete_actor100 import SACPolicyDiscrete100


__all__ = [
    "BasePolicy",
    "SACPolicy",
    "TD3Policy",
    "CQLPolicy",
    "IQLPolicy",
    "MCQPolicy",
    "CQLDQNPolicy",
    "TD3BCPolicy",
    "EDACPolicy",
    "MOPOPolicy",
    "MOBILEPolicy",
    "RAMBOPolicy",
    "COMBOPolicy",
    
    "SACTTAPolicy",
    "SACTTAMULTULPolicy",
    "MCQTTAPolicy",
    "CQLTTAPolicy",
    "IQLTTAPolicy",
    "MOPOTTAPolicy",
    "MCQRDropoutPolicy",
    "MCQTTABUFFERPolicy",
    "MCQTTAMULTULPolicy",
    "SACPolicyDiscrete",
    # "CQLPolicyDiscrete",
    "CQLPolicyDiscreteQ",
    "CQLPolicyDiscreteActor",
    "CQLPolicyDiscreteActor100",
    "SACPolicyDiscrete100"
]