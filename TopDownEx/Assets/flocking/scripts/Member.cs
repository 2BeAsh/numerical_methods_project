using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Member : MonoBehaviour
{
    public Vector3 position;
    public Vector3 velocity;
    public Vector3 acceleration;

    public Level level;
    public MemberConfig conf;

    Vector3 wanderTarget;

    Vector3 wanterTarget;
    private void Start()
    {
        level = FindObjectOfType<Level>();
        conf = FindObjectOfType<MemberConfig>();

        position = transform.position;
        velocity = new Vector3(0,0,0);
    }

    private void Update()
    {
        acceleration = Combine();
        acceleration = Vector3.ClampMagnitude(acceleration, conf.maxAcceleration); // Probably like np.clip()
        velocity = velocity + acceleration * Time.deltaTime;
        velocity = Vector3.ClampMagnitude(velocity, conf.maxVelocity);
        position = position + velocity * Time.deltaTime;
        WrapAround(ref position, -level.bounds, level.bounds);
        transform.position = position;
    }

    protected Vector3 Wander()
    {
        float jitter = conf.wanderJitter * Time.deltaTime;
        wanderTarget += new Vector3(RandomBinomial() * jitter, RandomBinomial() * jitter, 0);
        Vector3 targetInLocalSpace = wanterTarget + new Vector3(0, conf.wanderDistance, 0);
        Vector3 targetInWorldSpace = transform.TransformPoint(targetInLocalSpace);
        targetInWorldSpace -= this.position;

        return targetInWorldSpace.normalized;
    }

    Vector3 Cohesion()
    {
        Vector3 cohesionVector = new Vector3();
        int countMembers = 0;
        var neighbors = level.GetNeighbors(this, conf.cohesionRadius);
        if (neighbors.Count == 0) // Stop if no neigbors
        {
            return cohesionVector;
        }
        foreach (var member in neighbors)
        {
            if (isInFOV(member.position))
            {
                cohesionVector += member.position;
                countMembers++;
            }
        }
        if (countMembers == 0)
        {
            return cohesionVector;
        }

        cohesionVector /= countMembers;
        cohesionVector = cohesionVector - this.position;
        cohesionVector = Vector3.Normalize(cohesionVector);
        return cohesionVector; 
    }


    virtual protected Vector3 Combine()
    {
        Vector3 finalVec = conf.cohesionPriority * Cohesion() + conf.wanderPriority * Wander();
        return finalVec;
    }


    void WrapAround(ref Vector3 vector, float min, float max)
    {
        vector.x = BoundaryStop(vector.x, min, max);
        vector.y = BoundaryStop(vector.y, min, max);
        vector.z = BoundaryStop(vector.z, min, max);


    }
    // Create boundaries
    float BoundaryStop(float value, float min, float max)
    {
        if (value > max)
        {
            value = max * 0.99f;
        }
        else if (value < min)
        {
            value = min * 1.01f;
        }
        return value;
    }

    float RandomBinomial()
    {
        return Random.Range(0f, 1f) - Random.Range(0f, 1f);
    }

    bool isInFOV(Vector3 vec)
    {
        return Vector3.Angle(this.velocity, vec - this.position) <= conf.maxFOV;
    }




}
